import os
import torch
import torch.distributed as dist
import json
import csv
from tqdm import tqdm
from datetime import datetime
import math
from model_handler import QwenModel
from utils import setup_logger

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

class DistributedImageDescriptionGenerator:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.logger = setup_logger(f'GPU_{rank}')
        try:
            self.model = QwenModel(rank)
            self.logger.info(f"Successfully initialized model on GPU {rank}")
        except Exception as e:
            self.logger.error(f"Failed to initialize model on GPU {rank}: {str(e)}")
            raise

    def get_image_list(self, region_dir: str) -> list:
        image_paths = []
        for user_dir in os.listdir(region_dir):
            user_path = os.path.join(region_dir, user_dir)
            if not os.path.isdir(user_path):
                continue
            
            for image_file in os.listdir(user_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_paths.append((user_dir, os.path.join(user_path, image_file)))
        return image_paths

    def process_region_data(self, region_dir: str, csv_path: str, fieldnames: list):
        try:
            # 获取所有图片路径
            all_images = self.get_image_list(region_dir)
            
            # 计算当前GPU需要处理的图片
            images_per_gpu = math.ceil(len(all_images) / self.world_size)
            start_idx = self.rank * images_per_gpu
            end_idx = min(start_idx + images_per_gpu, len(all_images))
            
            # 获取当前GPU的图片子集
            gpu_images = all_images[start_idx:end_idx]
            
            self.logger.info(f"GPU {self.rank} processing {len(gpu_images)} images")
            
            # 创建临时CSV文件
            temp_csv_path = f"{csv_path}.part{self.rank}"
            with open(temp_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(fieldnames)
            
            processed_count = 0
            error_count = 0
            
            for user_id, image_path in tqdm(gpu_images, desc=f"GPU {self.rank} processing"):
                try:
                    result = self.model.describe_image(image_path)
                    
                    if result["status"] == "success":
                        data = result["result"]
                        row_data = [
                            user_id,
                            os.path.splitext(os.path.basename(image_path))[0],
                            data.get("实体", ""),
                            data.get("颜色", ""),
                            data.get("纹理", ""),
                            data.get("主题类型", ""),
                            data.get("情感", "")
                        ]
                        
                        with open(temp_csv_path, 'a', newline='', encoding='utf-8-sig') as f:
                            writer = csv.writer(f)
                            writer.writerow(row_data)
                        
                        processed_count += 1
                        if processed_count % 10 == 0:
                            self.logger.info(f"GPU {self.rank} processed {processed_count} images")
                    else:
                        error_count += 1
                        self.logger.error(f"Failed to process {image_path}: {result['message']}")
                        
                except Exception as e:
                    error_count += 1
                    self.logger.error(f"Error processing {image_path}: {str(e)}")
                    continue
            
            self.logger.info(f"GPU {self.rank} completed. Successful: {processed_count}, Errors: {error_count}")
            
            # 等待所有GPU完成处理
            dist.barrier()
            
            # 只在rank 0上合并结果
            if self.rank == 0:
                self.merge_results(csv_path, self.world_size)
                
        except Exception as e:
            self.logger.error(f"Error in process_region_data on GPU {self.rank}: {str(e)}")
            raise

    def merge_results(self, csv_path: str, world_size: int):
        """合并所有GPU的结果"""
        try:
            # 读取第一个文件的头部
            with open(f"{csv_path}.part0", 'r', encoding='utf-8-sig') as f:
                header = f.readline()
            
            # 创建最终的CSV文件
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as final_file:
                final_file.write(header)
                
                # 合并所有部分文件
                for i in range(world_size):
                    part_file = f"{csv_path}.part{i}"
                    with open(part_file, 'r', encoding='utf-8-sig') as f:
                        # 跳过头部
                        next(f)
                        # 复制内容
                        for line in f:
                            final_file.write(line)
                    # 删除部分文件
                    os.remove(part_file)
                    
            self.logger.info(f"Successfully merged results into {csv_path}")
            
        except Exception as e:
            self.logger.error(f"Error merging results: {str(e)}")
            raise

def process_dataset_distributed(rank, world_size, dataset_dir, output_dir):
    logger = setup_logger(f'Process_{rank}')
    generator = None
    
    try:
        # 初始化分布式环境
        setup(rank, world_size)
        logger.info(f"Process {rank}: Distributed environment initialized")
        
        # 初始化生成器
        generator = DistributedImageDescriptionGenerator(rank, world_size)
        logger.info(f"Process {rank}: Generator initialized")
        
        fieldnames = ["用户ID", "图片ID", "实体", "颜色", "纹理", "主题类型", "情感"]
        
        # 处理澳大利亚数据
        australia_dir = os.path.join(dataset_dir, 'combined_Australia')
        if os.path.exists(australia_dir):
            logger.info(f"GPU {rank} processing Australia data...")
            generator.process_region_data(
                australia_dir,
                os.path.join(output_dir, 'australia_results.csv'),
                fieldnames
            )
        
        # 处理中国数据
        china_dir = os.path.join(dataset_dir, 'combined_China')
        if os.path.exists(china_dir):
            logger.info(f"GPU {rank} processing China data...")
            generator.process_region_data(
                china_dir,
                os.path.join(output_dir, 'china_results.csv'),
                fieldnames
            )
    
    except Exception as e:
        if logger:
            logger.error(f"Error in process_dataset_distributed: {str(e)}")
        raise
    
    finally:
        if generator and hasattr(generator, 'logger'):
            generator.logger.info(f"Process {rank}: Cleaning up...")
        cleanup()