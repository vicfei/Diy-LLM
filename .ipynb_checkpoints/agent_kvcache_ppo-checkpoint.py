import os
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
# from langgraph.checkpoint import MemorySaver
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime

# ==================== LangGraph Agent状态定义 ====================
class AgentState(TypedDict):
    """Agent系统的状态定义"""
    messages: Annotated[List[Any], add_messages]
    original_poem: Optional[str]
    generated_poem: Optional[str]
    appreciation: Optional[str]
    revised_poem: Optional[str]
    revision_instruction: Optional[str]
    current_step: str
    generation_params: Dict[str, Any]

# ==================== 阿里云DashScope客户端 ====================
import dashscope
from dashscope import Generation

class DashScopeClient:
    """阿里云DashScope客户端"""
    
    def __init__(self, api_key: str):
        """初始化DashScope客户端"""
        dashscope.api_key = api_key
        self.model_analysis = "qwen-max"  # 诗词赏析模型
        self.model_revision = "qwen-max"  # 诗词修改模型
        
    def analyze_poetry(self, poem: str, style: str = "古典") -> str:
        """诗词赏析"""
        try:
            prompt = f"""请对以下{style}风格的诗词进行专业赏析，包括：
            1. 诗词的整体意境和主题
            2. 使用的修辞手法和表现技巧
            3. 语言特点和韵律美
            4. 蕴含的情感和思想深度
            
            诗词：
            {poem}
            
            请提供详细、专业的赏析："""
            
            response = Generation.call(
                model=self.model_analysis,
                prompt=prompt,
                max_tokens=1000,
                temperature=0.7
            )
            
            if response.status_code == 200:
                return response.output.text
            else:
                return f"赏析失败: {response.message}"
                
        except Exception as e:
            return f"赏析过程中出现错误: {str(e)}"
    
    def revise_poetry(self, poem: str, instruction: str) -> str:
        """诗词修改"""
        try:
            prompt = f"""请根据以下要求修改这首诗词：
            
            原始诗词：
            {poem}
            
            修改要求：{instruction}
            
            请提供修改后的完整诗词，并简要说明修改的地方："""
            
            response = Generation.call(
                model=self.model_revision,
                prompt=prompt,
                max_tokens=1000,
                temperature=0.7
            )
            
            if response.status_code == 200:
                return response.output.text
            else:
                return f"修改失败: {response.message}"
                
        except Exception as e:
            return f"修改过程中出现错误: {str(e)}"
    
    def evaluate_poetry(self, poem1: str, poem2: str, criteria: str = "意境、韵律、语言") -> str:
        """诗词对比评价"""
        try:
            prompt = f"""请从{criteria}等方面对比评价以下两首诗词：
            
            诗词A：
            {poem1}
            
            诗词B：
            {poem2}
            
            请给出详细的对比分析和评价："""
            
            response = Generation.call(
                model=self.model_analysis,
                prompt=prompt,
                max_tokens=800,
                temperature=0.7
            )
            
            if response.status_code == 200:
                return response.output.text
            else:
                return f"评价失败: {response.message}"
                
        except Exception as e:
            return f"评价过程中出现错误: {str(e)}"

# ==================== 诗词生成Agent ====================
class PoetryGenerationAgent:
    """诗词生成Agent，使用用户提供的模型"""
    
    def __init__(self, generator):
        """初始化诗词生成Agent"""
        self.generator = generator
    
    def generate(self, state: AgentState) -> AgentState:
        """生成诗词"""
        messages = state["messages"]
        
        # 提取用户输入
        user_input = ""
        for msg in messages[-3:]:  # 查看最近的消息
            if isinstance(msg, HumanMessage):
                user_input = msg.content
                break
        
        # 如果没有明确的提示，使用默认提示
        if not user_input or len(user_input.strip()) < 2:
            prompts = ["春江花月夜", "秋风起兮白云飞", "山水有相逢", "明月几时有"]
            import random
            prompt = random.choice(prompts)
        else:
            # 提取可能的提示词
            # 用户可能输入完整的句子，我们需要提取关键词
            import re
            # 尝试提取2-4个字的词语作为提示
            words = re.findall(r'[\u4e00-\u9fff]{2,4}', user_input)
            prompt = words[0] if words else user_input[:2]
        
        # 获取生成参数
        params = state.get("generation_params", {})
        max_len = params.get("max_len", 40)
        temperature = params.get("temperature", 0.7)
        
        # 生成诗词
        try:
            generated_poem = self.generator.generate_with_kv_cache(
                prompt=prompt,
                max_len=max_len,
                temperature=temperature
            )
            
            # 清理诗词格式
            generated_poem = self._clean_poem(generated_poem)
            
            # 更新状态
            state["generated_poem"] = generated_poem
            state["original_poem"] = generated_poem  # 初始时原始诗词就是生成的诗词
            state["current_step"] = "generation_completed"
            
            # 添加AI消息
            state["messages"].append(AIMessage(
                content=f"已为您生成一首诗词：\n\n{generated_poem}\n\n"
                       f"提示词：'{prompt}'，温度：{temperature}，最大长度：{max_len}"
            ))
            
        except Exception as e:
            state["messages"].append(AIMessage(
                content=f"诗词生成失败：{str(e)}"
            ))
            state["current_step"] = "generation_failed"
        
        return state
    
    def _clean_poem(self, poem: str) -> str:
        """清理诗词格式"""
        # 移除多余的空格和换行
        poem = poem.strip()
        
        # 确保每行不超过8个字符（传统诗词格式）
        lines = poem.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # 如果一行太长，尝试按标点分割
                if len(line) > 12:
                    # 尝试找标点分割
                    for punct in ['，', '。', '、', '；', '！', '？']:
                        if punct in line:
                            parts = line.split(punct)
                            for part in parts:
                                if part.strip():
                                    cleaned_lines.append(part.strip() + punct)
                            break
                    else:
                        # 没有标点，按固定长度分割
                        for i in range(0, len(line), 8):
                            cleaned_lines.append(line[i:i+8])
                else:
                    cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

# ==================== 诗词赏析Agent ====================
class PoetryAppreciationAgent:
    """诗词赏析Agent，使用阿里云API"""
    
    def __init__(self, dashscope_client):
        self.client = dashscope_client
    
    def appreciate(self, state: AgentState) -> AgentState:
        """赏析诗词"""
        poem = state.get("generated_poem") or state.get("original_poem")
        
        if not poem:
            state["messages"].append(AIMessage(
                content="没有找到需要赏析的诗词，请先生成一首诗词。"
            ))
            state["current_step"] = "appreciation_failed"
            return state
        
        try:
            # 获取诗词风格
            style = self._detect_poetry_style(poem)
            
            # 调用阿里云API进行赏析
            appreciation = self.client.analyze_poetry(poem, style)
            
            # 更新状态
            state["appreciation"] = appreciation
            state["current_step"] = "appreciation_completed"
            
            # 添加AI消息
            state["messages"].append(AIMessage(
                content=f"【诗词赏析】\n\n"
                       f"诗词风格：{style}\n\n"
                       f"{appreciation}"
            ))
            
        except Exception as e:
            state["messages"].append(AIMessage(
                content=f"诗词赏析失败：{str(e)}"
            ))
            state["current_step"] = "appreciation_failed"
        
        return state
    
    def _detect_poetry_style(self, poem: str) -> str:
        """检测诗词风格"""
        # 简单的风格检测逻辑
        if any(word in poem for word in ['春', '花', '月', '风', '柳']):
            return "婉约"
        elif any(word in poem for word in ['山', '水', '云', '天', '海']):
            return "山水田园"
        elif any(word in poem for word in ['酒', '剑', '马', '战', '沙']):
            return "豪放"
        elif any(word in poem for word in ['秋', '夜', '愁', '思', '泪']):
            return "抒情"
        else:
            return "古典"

# ==================== 诗词修改Agent ====================
class PoetryRevisionAgent:
    """诗词修改Agent，使用阿里云API"""
    
    def __init__(self, dashscope_client):
        self.client = dashscope_client
    
    def revise(self, state: AgentState) -> AgentState:
        """修改诗词"""
        poem = state.get("generated_poem") or state.get("original_poem")
        instruction = state.get("revision_instruction", "")
        
        if not poem:
            state["messages"].append(AIMessage(
                content="没有找到需要修改的诗词，请先生成一首诗词。"
            ))
            state["current_step"] = "revision_failed"
            return state
        
        # 如果没有修改指令，生成一个默认的
        if not instruction:
            instruction = "请优化诗词的韵律和意境，使其更加优美"
        
        try:
            # 调用阿里云API进行修改
            revised_result = self.client.revise_poetry(poem, instruction)
            
            # 尝试从结果中提取修改后的诗词
            revised_poem = self._extract_revised_poem(revised_result, poem)
            
            # 更新状态
            state["revised_poem"] = revised_poem
            state["current_step"] = "revision_completed"
            
            # 添加AI消息
            state["messages"].append(AIMessage(
                content=f"【诗词修改】\n\n"
                       f"修改要求：{instruction}\n\n"
                       f"修改结果：\n{revised_result}"
            ))
            
        except Exception as e:
            state["messages"].append(AIMessage(
                content=f"诗词修改失败：{str(e)}"
            ))
            state["current_step"] = "revision_failed"
        
        return state
    
    def _extract_revised_poem(self, revised_result: str, original_poem: str) -> str:
        """从修改结果中提取诗词"""
        # 尝试找到修改后的诗词部分
        lines = revised_result.split('\n')
        
        # 查找可能的诗词开始
        for i, line in enumerate(lines):
            if any(marker in line for marker in ['修改后', '修改版', '改后', '新版']):
                # 从下一行开始提取
                poem_lines = []
                for j in range(i + 1, min(i + 20, len(lines))):
                    if lines[j].strip() and not lines[j].startswith('说明'):
                        poem_lines.append(lines[j].strip())
                if poem_lines:
                    return '\n'.join(poem_lines)
        
        # 如果没有找到，返回原始结果的前10行
        return '\n'.join(lines[:10]) if len(lines) > 10 else revised_result

# ==================== 路由逻辑 ====================
class Router:
    """路由决策类"""
    
    @staticmethod
    def should_appreciate(state: AgentState) -> str:
        """决定是否进行赏析"""
        messages = state["messages"]
        
        # 检查用户是否要求赏析
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                content = msg.content.lower()
                if any(word in content for word in ['赏析', '分析', '评价', '欣赏', '解读']):
                    return "yes"
                elif any(word in content for word in ['不用赏析', '不赏析', '跳过赏析']):
                    return "no"
        
        # 如果已经有生成的诗词且没有明确的指示，默认进行赏析
        if state.get("generated_poem") and not state.get("appreciation"):
            return "yes"
        
        return "no"
    
    @staticmethod
    def should_revise(state: AgentState) -> str:
        """决定是否进行修改"""
        messages = state["messages"]
        
        # 检查用户是否要求修改
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                content = msg.content.lower()
                if any(word in content for word in ['修改', '改写', '调整', '优化', '改进']):
                    # 提取修改指令
                    import re
                    # 尝试提取用户的具体要求
                    if '修改' in content or '改写' in content:
                        # 提取修改指令
                        instruction = content.replace('修改', '').replace('改写', '').strip()
                        if instruction and len(instruction) > 2:
                            state["revision_instruction"] = instruction
                    return "yes"
                elif any(word in content for word in ['不用修改', '不修改', '跳过修改']):
                    return "no"
        
        # 检查是否已经要求过修改
        if state.get("revision_instruction"):
            return "yes"
        
        return "no"

# ==================== 主Agent系统 ====================
class PoetryMultiAgentSystem:
    """多Agent诗词系统"""
    
    def __init__(self, model_path: str, vocab_file: str, dashscope_api_key: str):
        """初始化多Agent系统"""
        print("正在初始化多Agent诗词系统...")
        
        # 初始化诗词生成器
        print("正在加载诗词生成模型...")
        self.generator = self._initialize_generator(model_path, vocab_file)
        
        # 初始化阿里云客户端
        print("正在初始化阿里云DashScope客户端...")
        self.dashscope_client = DashScopeClient(dashscope_api_key)
        
        # 初始化各个Agent
        print("正在初始化各个Agent...")
        self.generation_agent = PoetryGenerationAgent(self.generator)
        self.appreciation_agent = PoetryAppreciationAgent(self.dashscope_client)
        self.revision_agent = PoetryRevisionAgent(self.dashscope_client)
        
        # 构建工作流
        print("正在构建工作流...")
        self.workflow = self._build_workflow()
        
        print("多Agent诗词系统初始化完成！")
    
    def _initialize_generator(self, model_path: str, vocab_file: str):
        """初始化诗词生成器"""
        import torch
        
        # 检查文件是否存在
        import os
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"词汇表文件不存在: {vocab_file}")
        
        # 使用用户提供的模型代码
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        generator = KVCachePoetryGenerator(
            model_path=model_path,
            vocab_file=vocab_file,
            device=device,
            use_half=(device.type == 'cuda'),  # GPU上使用半精度
            max_batch_size=4
        )
        
        return generator
    
    def _build_workflow(self):
        """构建LangGraph工作流"""
        # 创建状态图
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("generation", self.generation_agent.generate)
        workflow.add_node("appreciation", self.appreciation_agent.appreciate)
        workflow.add_node("revision", self.revision_agent.revise)
        workflow.add_node("final_response", self._final_response)
        
        # 设置入口点
        workflow.set_entry_point("generation")
        
        # 从生成节点到条件路由
        workflow.add_conditional_edges(
            "generation",
            self._route_after_generation,
            {
                "appreciate": "appreciation",
                "revise": "revision",
                "end": "final_response"
            }
        )
        
        # 从赏析节点到条件路由
        workflow.add_conditional_edges(
            "appreciation",
            self._route_after_appreciation,
            {
                "revise": "revision",
                "end": "final_response"
            }
        )
        
        # 从修改节点到结束
        workflow.add_edge("revision", "final_response")
        
        # 从最终响应节点到结束
        workflow.add_edge("final_response", END)
        
        # 添加记忆存储
        memory = MemorySaver()
        
        # 编译工作流
        return workflow.compile(checkpointer=memory)
    
    def _route_after_generation(self, state: AgentState) -> str:
        """生成后的路由逻辑"""
        # 检查是否需要赏析
        if Router.should_appreciate(state) == "yes":
            return "appreciate"
        
        # 检查是否需要修改
        if Router.should_revise(state) == "yes":
            return "revise"
        
        return "end"
    
    def _route_after_appreciation(self, state: AgentState) -> str:
        """赏析后的路由逻辑"""
        # 检查是否需要修改
        if Router.should_revise(state) == "yes":
            return "revise"
        
        return "end"
    
    def _final_response(self, state: AgentState) -> AgentState:
        """生成最终响应"""
        messages = state["messages"]
        
        # 汇总结果
        summary = "【诗词创作完成】\n\n"
        
        if state.get("generated_poem"):
            summary += f"生成的诗词：\n{state['generated_poem']}\n\n"
        
        if state.get("appreciation"):
            summary += f"诗词赏析：\n{state['appreciation'][:300]}...\n\n"
        
        if state.get("revised_poem"):
            summary += f"修改后的诗词：\n{state['revised_poem']}\n\n"
        
        summary += "您可以选择：\n"
        summary += "1. 输入新的提示词生成另一首诗词\n"
        summary += "2. 输入'赏析'对当前诗词进行赏析\n"
        summary += "3. 输入'修改 [要求]'对当前诗词进行修改\n"
        summary += "4. 输入'结束'退出系统"
        
        # 添加最终消息
        state["messages"].append(AIMessage(content=summary))
        state["current_step"] = "completed"
        
        return state
    
    def process_request(self, user_input: str, config: Dict[str, Any] = None) -> str:
        """处理用户请求"""
        # 初始化状态
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "original_poem": None,
            "generated_poem": None,
            "appreciation": None,
            "revised_poem": None,
            "revision_instruction": None,
            "current_step": "start",
            "generation_params": config or {
                "max_len": 40,
                "temperature": 0.7
            }
        }
        
        try:
            # 执行工作流
            final_state = self.workflow.invoke(initial_state, config={"configurable": {"thread_id": "1"}})
            
            # 获取最后的AI消息
            messages = final_state["messages"]
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    return msg.content
            
            return "处理完成，但未生成响应。"
            
        except Exception as e:
            return f"处理请求时出错: {str(e)}"
    
    def interactive_mode(self):
        """交互式模式"""
        print("\n" + "="*60)
        print("多Agent诗词系统 - 交互模式")
        print("="*60)
        print("请输入您的请求（输入'结束'退出）：")
        print("- 直接输入主题词生成诗词，如：'春天'")
        print("- 输入'赏析'对当前诗词进行赏析")
        print("- 输入'修改 [要求]'对当前诗词进行修改")
        print("- 输入'设置 参数名=值'设置生成参数")
        print("="*60)
        
        config = {"max_len": 40, "temperature": 0.7}
        
        while True:
            try:
                user_input = input("\n您: ").strip()
                
                if user_input.lower() in ['结束', 'exit', 'quit', 'q']:
                    print("感谢使用，再见！")
                    break
                
                elif user_input.lower() == '帮助' or user_input == '?':
                    print("可用命令：")
                    print("  [主题词] - 生成相关主题的诗词")
                    print("  赏析 - 对当前诗词进行赏析")
                    print("  修改 [要求] - 修改当前诗词")
                    print("  设置 max_len=50 - 设置生成的最大长度")
                    print("  设置 temperature=0.8 - 设置生成温度")
                    print("  结束 - 退出系统")
                    continue
                
                elif user_input.startswith('设置 '):
                    # 处理参数设置
                    setting = user_input[3:].strip()
                    if '=' in setting:
                        key, value = setting.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        if key in ['max_len', 'temperature']:
                            try:
                                if key == 'max_len':
                                    config[key] = int(value)
                                else:
                                    config[key] = float(value)
                                print(f"已设置 {key} = {value}")
                            except ValueError:
                                print(f"无效的值: {value}")
                        else:
                            print(f"未知参数: {key}")
                    continue
                
                # 处理请求
                print("\n" + "-"*40)
                print("系统处理中...")
                response = self.process_request(user_input, config)
                print("\n" + "="*40)
                print(response)
                print("="*40)
                
            except KeyboardInterrupt:
                print("\n\n检测到中断，正在退出...")
                break
            except Exception as e:
                print(f"\n处理时出错: {str(e)}")

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 配置参数
    MODEL_PATH = "stable_rl_model.pth"  # 您的模型路径
    VOCAB_FILE = "vocab.txt"           # 您的词汇表路径
    DASHSCOPE_API_KEY = "[REDACTED]"  # 您的API密钥
    
    try:
        # 创建多Agent系统
        system = PoetryMultiAgentSystem(
            model_path=MODEL_PATH,
            vocab_file=VOCAB_FILE,
            dashscope_api_key=DASHSCOPE_API_KEY
        )
        
        # 启动交互模式
        system.interactive_mode()
        
    except FileNotFoundError as e:
        print(f"文件错误: {e}")
        print("请确保以下文件存在:")
        print(f"1. 模型文件: {MODEL_PATH}")
        print(f"2. 词汇表文件: {VOCAB_FILE}")
    
    except Exception as e:
        print(f"系统初始化失败: {str(e)}")
        print("\n可能的解决方案:")
        print("1. 检查模型文件和词汇表文件路径是否正确")
        print("2. 检查阿里云API密钥是否正确")
        print("3. 确保已安装所有依赖包: pip install langgraph langchain-core dashscope")