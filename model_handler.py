import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import json
import re
import logging
import os
class QwenModel:
    def __init__(self, rank):
        self.rank = rank
        self.logger = logging.getLogger(f'QwenModel_GPU_{rank}')
        self.init_model()
        self.prompt = """你现在是一个图片内容理解与文本标注专家，现在需要进行图片内容理解任务，告诉我图片的视觉描述性文本。
                    我会给你一系列旅游景点图片数据作为输入,请你用文字分别描述一下每个旅游景点图片中的实体、颜色、纹理、主题类型、情感这五个方面的短文本信息。每类不超过20个字。
                    其中,实体是指图像中所代表的旅游景点或项目主体(如：高山、游乐园、大海、沙滩、都市等),颜色是指图像中呈现的主要色彩,纹理是指图像表面的质感或纹理特征,主题类型是指图像所代表的旅游主题和类型（如：文化旅游、生态旅游、冒险旅游、医疗旅游、宗教旅游、乡村旅游、背包旅游等），情感是指图像传达出的情感氛围（如：欢乐的、放松的、刺激的、浪漫的、怀旧的、古典的等），你可以基于这些理智思考总结与扩展：
                    对于不同图片，每一条输出格式为：
                    Result:
                    {
                        "photo_id":[]
                            {
                                "实体": "描述图片中的主要实体,不超过10字",
                                "颜色": "描述主要颜色,不超过10字",
                                "纹理": "描述纹理特征,不超过10字",
                                "主题类型": "描述图片主题类型,不超过10字",
                                "情感": "描述图片传达的情感,不超过10字"
                            }
                    }
                    请严格按照json格式输出, 以确保后续解析正确"""

    def init_model(self):
        try:
            model_dir = "/root/workspace/Qwen_Model/Qwen/Qwen-VL-Chat"
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            
            # 修改模型加载参数
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                device_map=f"cuda:{self.rank}",  # 明确指定GPU设备
                torch_dtype=torch.float16,  # 使用 float16 来节省显存
                trust_remote_code=True
            ).eval()
            
            self.model.generation_config = GenerationConfig.from_pretrained(
                model_dir,
                trust_remote_code=True
            )
            
            self.logger.info(f"Successfully loaded model on GPU {self.rank}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model on GPU {self.rank}: {str(e)}")
            raise

    def describe_image(self, image_path: str) -> dict:
        try:
            if not os.path.exists(image_path):
                self.logger.error(f"Image file not found: {image_path}")
                return {"status": "error", "message": "File does not exist"}

            query = self.tokenizer.from_list_format([
                {'image': image_path},
                {'text': self.prompt}
            ])

            response, _ = self.model.chat(self.tokenizer, query=query, history=None)
            self.logger.debug(f"Model response: {response}")
            
            try:
                # 尝试修复缺失的右花括号
                if response.count('{') > response.count('}'):
                    response = response + '}'
                
                # 尝试清理响应文本
                response = response.strip()
                if not response.endswith('}'):
                    last_brace = response.rfind('}')
                    if last_brace != -1:
                        response = response[:last_brace+1]
                
                # 尝试解析JSON
                try:
                    result = json.loads(response)
                except json.JSONDecodeError:
                    # 如果解析失败，尝试用正则表达式提取字段
                    fields = {}
                    patterns = {
                        "实体": r'"实体":\s*"([^"]*)"',
                        "颜色": r'"颜色":\s*"([^"]*)"',
                        "纹理": r'"纹理":\s*"([^"]*)"',
                        "主题类型": r'"主题类型":\s*"([^"]*)"',
                        "情感": r'"情感":\s*"([^"]*)"'
                    }
                    
                    for field, pattern in patterns.items():
                        match = re.search(pattern, response)
                        if match:
                            fields[field] = match.group(1)
                    
                    if fields:
                        result = fields
                    else:
                        raise ValueError("Could not extract fields from response")

                # 验证结果格式
                required_fields = ["实体", "颜色", "纹理", "主题类型", "情感"]
                if not all(field in result for field in required_fields):
                    missing_fields = [field for field in required_fields if field not in result]
                    self.logger.warning(f"Missing fields in response: {missing_fields}")
                    # 为缺失字段添加空值
                    for field in missing_fields:
                        result[field] = ""
                
                return {"status": "success", "result": result}
                
            except Exception as e:
                self.logger.error(f"Failed to parse response: {response}")
                self.logger.error(f"Error details: {str(e)}")
                # 尝试从错误的响应中提取有用信息
                try:
                    partial_result = {
                        "实体": "",
                        "颜色": "",
                        "纹理": "",
                        "主题类型": "",
                        "情感": ""
                    }
                    
                    lines = response.split('\n')
                    for line in lines:
                        for field in partial_result.keys():
                            if f'"{field}":' in line:
                                try:
                                    value = line.split(':')[1].strip().strip('"').strip(',').strip()
                                    partial_result[field] = value
                                except:
                                    continue
                    
                    return {"status": "success", "result": partial_result}
                except:
                    return {"status": "error", "message": f"Invalid response format: {str(e)}"}

        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {str(e)}")
            return {"status": "error", "message": str(e)}