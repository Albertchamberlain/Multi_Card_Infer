import os
import logging
import torch.multiprocessing as mp
from datetime import datetime
from distributed_generator import process_dataset_distributed
import torch

def setup_main_logger():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/main_{timestamp}.log'
    
    logger = logging.getLogger('Main')
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def main():
    logger = setup_main_logger()
    
    try:
        # 配置路径
        dataset_dir = "/root/workspace/dataset"
        output_dir = "/root/workspace/output"
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取可用的GPU数量
        world_size = torch.cuda.device_count()
        
        if world_size == 0:
            raise RuntimeError("No GPU devices available")
        
        logger.info(f"Found {world_size} GPU devices")
        
        # 设置多进程方法为 spawn
        mp.set_start_method('spawn', force=True)
        
        # 启动多GPU处理
        mp.spawn(
            process_dataset_distributed,
            args=(world_size, dataset_dir, output_dir),
            nprocs=world_size,
            join=True
        )
        
        logger.info(f"Processing completed. Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()