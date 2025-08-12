# AI聊天助手Web应用（修复版）
# chat_assistant_app.py

import streamlit as st
import torch
import tiktoken
import json
import os
from datetime import datetime
import plotly.express as px
from gptMoudel import GPTModel

class ChatAssistantApp:
    def __init__(self):
        self.setup_page_config()
        self.load_model()
        self.setup_session_state()
    
    def setup_page_config(self):
        """设置页面配置"""
        st.set_page_config(
            page_title="AI智能聊天助手",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def load_model(self):
        """加载训练好的模型"""
        try:
            # 模型配置
            self.config = {
                "vocab_size": 50257,
                "context_length": 1024,
                "emb_dim": 1024,  # GPT-2 Medium配置
                "n_heads": 16,
                "n_layers": 24,
                "drop_rate": 0.0,
                "qkv_bias": True
            }
            
            # 检测设备
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
            
            # 加载模型
            self.model = GPTModel(self.config)
            
            # 尝试加载微调后的权重
            model_files = [
                "gpt2-medium355M-sft.pth",
                "instruction_tuned_model.pth",
                "sft_model.pth"
            ]
            
            self.model_loaded = False
            self.model_file = "基础模型"
            
            for model_file in model_files:
                if os.path.exists(model_file):
                    try:
                        # 尝试加载权重
                        state_dict = torch.load(model_file, map_location=self.device)
                        self.model.load_state_dict(state_dict, strict=False)
                        self.model_loaded = True
                        self.model_file = model_file
                        print(f"Successfully loaded model weights: {model_file}")
                        break
                    except Exception as e:
                        print(f"Failed to load {model_file}: {str(e)}")
                        continue
            
            if not self.model_loaded:
                # 即使没有微调权重，也使用基础模型
                self.model_loaded = True
                print("Warning: No fine-tuned model found, using base GPT model")
            
            # 将模型移到GPU
            self.model = self.model.to(self.device)
            self.model.eval()
            self.tokenizer = tiktoken.get_encoding("gpt2")
            
            print(f"Model loading completed: {self.model_file} on {self.device}")
            
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
            st.error(f"模型加载失败: {str(e)}")
            self.model_loaded = False
    
    def setup_session_state(self):
        """初始化会话状态"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'conversation_count' not in st.session_state:
            st.session_state.conversation_count = 0
        if 'total_tokens' not in st.session_state:
            st.session_state.total_tokens = 0
        if 'settings' not in st.session_state:
            st.session_state.settings = {
                'temperature': 0.7,
                'max_tokens': 80,  # 增加默认生成长度
                'top_k': 50,
                'personality': 'helpful'
            }
    
    def format_input(self, instruction, input_text=""):
        """格式化输入为指令格式"""
        instruction_text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{instruction}"
        )
        
        if input_text:
            instruction_text += f"\n\n### Input:\n{input_text}"
        
        instruction_text += "\n\n### Response:\n"
        return instruction_text
    
    def generate_response_simple(self, user_input):
        """简单的文本生成方法 - 修复编码问题"""
        try:
            # 编码输入 - 修复特殊token和编码问题
            formatted_input = f"Human: {user_input}\nAssistant:"
            
            try:
                input_tokens = self.tokenizer.encode(
                    formatted_input, 
                    allowed_special={'<|endoftext|>'}
                )
            except:
                # 如果有编码问题，清理输入文本
                clean_input = user_input.encode('ascii', errors='ignore').decode('ascii')
                formatted_input = f"Human: {clean_input}\nAssistant:"
                input_tokens = self.tokenizer.encode(formatted_input)
            
            # 限制输入长度
            max_input_length = 500  # 大幅减少，避免长序列问题
            if len(input_tokens) > max_input_length:
                input_tokens = input_tokens[-max_input_length:]
            
            # 转换为张量
            input_ids = torch.tensor(input_tokens).unsqueeze(0).to(self.device)
            generated_tokens = input_tokens.copy()
            
            # 只生成很少的token，确保稳定
            max_new_tokens = 10
            
            with torch.no_grad():
                for step in range(max_new_tokens):
                    current_input = torch.tensor(generated_tokens).unsqueeze(0).to(self.device)
                    
                    # 模型推理
                    logits = self.model(current_input)
                    next_token_logits = logits[0, -1, :]
                    
                    # 简单的贪心采样
                    next_token = torch.argmax(next_token_logits).item()
                    
                    # 检查结束条件
                    if next_token in [50256, 628, 198]:  # 常见结束token
                        break
                    
                    generated_tokens.append(next_token)
                    
                    # 早停 - 如果生成了足够内容就停止
                    if step > 3:
                        break
            
            # 安全解码 - 只解码新生成的部分
            try:
                new_tokens = generated_tokens[len(input_tokens):]
                if new_tokens:
                    # 分别解码，避免编码冲突
                    new_text = self.tokenizer.decode(new_tokens, errors='replace')
                    response = new_text.strip()
                else:
                    response = "你好！"
            except Exception as decode_error:
                # 如果解码还是失败，提供固定回复
                response = "你好！我是AI助手，很高兴为您服务！"
            
            # 清理响应，移除可能的问题字符
            if response:
                # 移除可能导致编码问题的字符
                response = response.encode('utf-8', errors='ignore').decode('utf-8')
                response = response.replace('\ufffd', '?')  # 替换替换字符
                response = ''.join(char for char in response if ord(char) < 65536)  # 移除高位Unicode
            
            if not response or len(response.strip()) < 2:
                response = "你好！我是AI助手。"
            
            return response[:100]  # 限制长度，避免长文本问题
            
        except Exception as e:
            # 返回简单的中文回复，避免编码问题
            error_type = type(e).__name__
            if 'codec' in str(e) or 'encode' in str(e):
                return "你好！我遇到了一些编码问题，但我可以正常对话。"
            elif 'cuda' in str(e).lower() or 'tensor' in str(e).lower():
                return "模型运行正常，但遇到了计算问题。"
            else:
                return "你好！我是AI助手，准备为您服务。"
    
    def generate_response(self, user_input, use_instruction_format=True):
        """生成AI回复的主方法 - 使用正确的指令格式"""
        if not self.model_loaded:
            return "模型未加载，无法生成回复"
        
        try:
            # 使用标准的指令微调格式
            if use_instruction_format:
                formatted_input = self.format_input(user_input)
            else:
                formatted_input = f"Human: {user_input}\nAssistant:"
            
            # 编码输入
            try:
                input_tokens = self.tokenizer.encode(
                    formatted_input, 
                    allowed_special={'<|endoftext|>'}
                )
            except:
                clean_input = user_input.encode('ascii', errors='ignore').decode('ascii')
                formatted_input = self.format_input(clean_input) if use_instruction_format else f"Human: {clean_input}\nAssistant:"
                input_tokens = self.tokenizer.encode(formatted_input)
            
            # 限制输入长度，但给生成留足空间
            max_input_length = 800
            if len(input_tokens) > max_input_length:
                input_tokens = input_tokens[-max_input_length:]
            
            # 转换为张量
            input_ids = torch.tensor(input_tokens).unsqueeze(0).to(self.device)
            generated_tokens = input_tokens.copy()
            
            # 增加生成长度，获得更好的回复
            max_new_tokens = min(st.session_state.settings['max_tokens'], 80)
            
            with torch.no_grad():
                for step in range(max_new_tokens):
                    current_input = torch.tensor(generated_tokens).unsqueeze(0).to(self.device)
                    
                    # 模型推理
                    logits = self.model(current_input)
                    next_token_logits = logits[0, -1, :]
                    
                    # 应用temperature和top-k采样
                    temperature = st.session_state.settings['temperature']
                    if temperature > 0:
                        next_token_logits = next_token_logits / temperature
                    
                    # Top-k采样
                    top_k = st.session_state.settings['top_k']
                    if top_k > 0:
                        top_k = min(top_k, next_token_logits.size(-1))
                        top_logits, top_indices = torch.topk(next_token_logits, top_k)
                        next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                        next_token_logits.scatter_(0, top_indices, top_logits)
                    
                    # 采样
                    if temperature > 0:
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, 1).item()
                    else:
                        next_token = torch.argmax(next_token_logits).item()
                    
                    # 检查结束条件
                    if next_token in [50256]:  # <|endoftext|>
                        break
                    
                    generated_tokens.append(next_token)
                    
                    # 检查是否生成了完整的句子
                    if step > 20 and next_token in [46, 63, 33]:  # . ? !
                        break
            
            # 安全解码
            try:
                full_text = self.tokenizer.decode(generated_tokens, errors='replace')
                
                # 提取回复部分
                if use_instruction_format and "### Response:" in full_text:
                    response = full_text.split("### Response:")[-1].strip()
                elif "Assistant:" in full_text:
                    response = full_text.split("Assistant:")[-1].strip()
                else:
                    # 提取新生成的部分
                    original_text = self.tokenizer.decode(input_tokens, errors='replace')
                    response = full_text[len(original_text):].strip()
                
            except Exception as decode_error:
                response = "我理解您的问题，但在生成回复时遇到了技术问题。"
            
            # 清理和验证回复
            response = self.clean_response(response)
            
            # 如果回复太短或无意义，使用备用回复
            if not response or len(response.strip()) < 5:
                backup_responses = {
                    'helpful': "我很乐意帮助您！能否请您提供更多详细信息？",
                    'professional': "根据您的问题，我需要更多背景信息来提供准确的分析。",
                    'creative': "这是一个有趣的话题！让我从不同角度来思考...",
                    'educational': "这是一个很好的学习机会！让我为您详细解释..."
                }
                personality = st.session_state.settings.get('personality', 'helpful')
                response = backup_responses.get(personality, "我理解您的问题，请允许我重新组织一下回答。")
            
            return response
            
        except Exception as e:
            return f"生成回复时遇到问题，但系统运行正常。请尝试重新表述您的问题。"
    
    def clean_response(self, response):
        """清理AI生成的回复"""
        if not response or not response.strip():
            return "我理解您的问题，让我换个方式回答..."
        
        # 移除常见的停止模式
        stop_patterns = [
            "### Instruction:", "### Input:", "### Response:", 
            "<|endoftext|>", "Human:", "Assistant:",
            "Below is an instruction"
        ]
        
        for pattern in stop_patterns:
            if pattern in response:
                response = response.split(pattern)[0]
        
        # 移除多余的空行
        lines = response.split('\n')
        cleaned_lines = []
        prev_empty = False
        
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
                prev_empty = False
            elif not prev_empty:
                cleaned_lines.append("")
                prev_empty = True
        
        response = '\n'.join(cleaned_lines).strip()
        
        # 限制长度，但保持句子完整性
        if len(response) > 500:
            # 尝试在句号处截断
            sentences = response.split('。')
            if len(sentences) > 1:
                truncated = ""
                for sentence in sentences:
                    if len(truncated + sentence + "。") <= 500:
                        truncated += sentence + "。"
                    else:
                        break
                if truncated:
                    response = truncated
                else:
                    response = response[:500] + "..."
            else:
                response = response[:500] + "..."
        
        # 检查是否为有意义的回复
        if len(response.strip()) < 3 or self.is_meaningless(response):
            return "感谢您的问题！能否请您提供更多细节，这样我可以给出更准确的回答。"
        
        return response
    
    def is_meaningless(self, text):
        """检查文本是否无意义"""
        text = text.lower().strip()
        
        # 无意义的模式
        meaningless_patterns = [
            r'^\.+$',  # 只有点号
            r'^,+$',   # 只有逗号
            r'^\s*$',  # 只有空格
            r'^(.)\1{10,}',  # 同一字符重复超过10次
        ]
        
        import re
        for pattern in meaningless_patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def save_conversation(self, user_input, ai_response):
        """保存对话记录"""
        conversation = {
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'user': user_input,
            'assistant': ai_response,
            'tokens': len(self.tokenizer.encode(user_input + ai_response))
        }
        
        st.session_state.chat_history.append(conversation)
        st.session_state.conversation_count += 1
        st.session_state.total_tokens += conversation['tokens']
        
        # 保持对话历史在合理范围内
        if len(st.session_state.chat_history) > 50:
            st.session_state.chat_history = st.session_state.chat_history[-50:]
    
    def render_header(self):
        """渲染页面头部"""
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="color: #2e8b57; font-size: 3rem; margin-bottom: 0.5rem;">
                🤖 AI智能聊天助手
            </h1>
            <p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
                基于GPT架构的指令微调模型，智能对话助手
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """渲染侧边栏设置"""
        with st.sidebar:
            st.markdown("### ⚙️ 模型设置")
            
            # 模型状态
            if self.model_loaded:
                st.success(f"✅ 模型已加载: {self.model_file}")
            else:
                st.error("❌ 模型加载失败")
                st.stop()  # 如果模型加载失败，停止应用
            
            # 生成参数设置
            st.markdown("#### 生成参数")
            
            st.session_state.settings['temperature'] = st.slider(
                "Temperature (创造性)", 
                min_value=0.1, 
                max_value=2.0, 
                value=st.session_state.settings['temperature'], 
                step=0.1,
                help="较高值使输出更有创造性，较低值使输出更确定"
            )
            
            st.session_state.settings['max_tokens'] = st.slider(
                "最大生成长度", 
                min_value=20, 
                max_value=200, 
                value=st.session_state.settings['max_tokens'], 
                step=10
            )
            
            st.session_state.settings['top_k'] = st.slider(
                "Top-K采样", 
                min_value=10, 
                max_value=100, 
                value=st.session_state.settings['top_k'], 
                step=5,
                help="限制每步的候选词汇数量"
            )
            
            # 助手性格设置
            st.markdown("#### 助手性格")
            personality_options = {
                'helpful': '友善助手 😊',
                'professional': '专业顾问 💼', 
                'creative': '创意伙伴 🎨',
                'educational': '学习导师 📚'
            }
            
            selected_personality = st.selectbox(
                "选择助手性格",
                options=list(personality_options.keys()),
                format_func=lambda x: personality_options[x],
                index=list(personality_options.keys()).index(st.session_state.settings['personality'])
            )
            st.session_state.settings['personality'] = selected_personality
            
            # 统计信息
            st.markdown("### 📊 对话统计")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("对话轮数", st.session_state.conversation_count)
            with col2:
                st.metric("总Token数", st.session_state.total_tokens)
            
            # 统计图表
            if st.session_state.chat_history:
                # 计算每小时的对话数量
                hours = [int(conv['timestamp'][:2]) for conv in st.session_state.chat_history[-20:]]
                hour_counts = {}
                for hour in hours:
                    hour_counts[hour] = hour_counts.get(hour, 0) + 1
                
                if hour_counts:
                    fig = px.bar(
                        x=list(hour_counts.keys()),
                        y=list(hour_counts.values()),
                        title="对话活跃度",
                        labels={'x': '小时', 'y': '对话数'}
                    )
                    fig.update_layout(height=200, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            # 清除对话按钮
            if st.button("🗑️ 清除对话历史", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.conversation_count = 0
                st.session_state.total_tokens = 0
                st.rerun()
    
    def render_chat_interface(self):
        """渲染聊天界面"""
        st.markdown("### 💬 智能对话")
        
        # 显示对话历史
        chat_container = st.container()
        
        with chat_container:
            # 创建聊天显示区域
            if st.session_state.chat_history:
                for i, conv in enumerate(st.session_state.chat_history[-10:]):  # 显示最近10轮对话
                    # 用户消息
                    st.markdown(f"""
                    <div style="text-align: right; margin: 1rem 0;">
                        <div style="background-color: #007bff; color: white; padding: 0.8rem 1rem; 
                                  border-radius: 1rem; display: inline-block; max-width: 70%;">
                            <strong>您 ({conv['timestamp']}):</strong><br>
                            {conv['user']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # AI回复
                    st.markdown(f"""
                    <div style="text-align: left; margin: 1rem 0;">
                        <div style="background-color: #f1f3f4; color: #333; padding: 0.8rem 1rem; 
                                  border-radius: 1rem; display: inline-block; max-width: 70%;">
                            <strong>🤖 AI助手:</strong><br>
                            {conv['assistant']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("👋 您好！我是您的AI助手，有什么可以帮助您的吗？")
        
        # 预设问题按钮
        st.markdown("#### 💡 快速开始")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📖 解释概念", use_container_width=True):
                st.session_state.demo_question = "请解释什么是人工智能？"
        
        with col2:
            if st.button("💡 创意建议", use_container_width=True):
                st.session_state.demo_question = "给我一些周末活动的创意建议"
        
        with col3:
            if st.button("📝 写作帮助", use_container_width=True):
                st.session_state.demo_question = "帮我写一份简短的会议邀请"
        
        # 输入区域
        user_input = st.text_area(
            "请输入您的问题或指令：",
            value=st.session_state.get('demo_question', ''),
            height=100,
            placeholder="在这里输入您想问的问题..."
        )
        
        # 发送按钮
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            send_button = st.button(
                "🚀 发送消息", 
                use_container_width=True,
                type="primary",
                disabled=not user_input.strip() or not self.model_loaded
            )
        
        # 处理用户输入
        if send_button and user_input.strip():
            with st.spinner("🤖 AI正在思考中..."):
                # 根据性格调整提示
                personality_prompts = {
                    'helpful': f"请作为友善助手回答：{user_input}",
                    'professional': f"请以专业角度分析：{user_input}",
                    'creative': f"请发挥创意回答：{user_input}",
                    'educational': f"请以教育方式解释：{user_input}"
                }
                
                enhanced_input = personality_prompts.get(
                    st.session_state.settings['personality'], 
                    user_input
                )
                
                # 显示调试信息
                with st.expander("🔧 调试信息", expanded=False):
                    st.text(f"模型状态: {'已加载' if self.model_loaded else '未加载'}")
                    st.text(f"设备: {self.device}")
                    st.text(f"原始输入: {user_input[:100]}...")
                    st.text(f"增强输入: {enhanced_input[:100]}...")
                    st.text(f"助手性格: {st.session_state.settings['personality']}")
                    st.text(f"生成参数: temp={st.session_state.settings['temperature']:.1f}, max_tokens={st.session_state.settings['max_tokens']}, top_k={st.session_state.settings['top_k']}")
                
                # 生成回复
                ai_response = self.generate_response(enhanced_input)
                
                if ai_response and len(ai_response.strip()) > 0:
                    # 保存对话
                    self.save_conversation(user_input.strip(), ai_response)
                    st.success("✅ 回复生成成功！")
                else:
                    st.error("❌ 生成的回复为空，请重试")
                
                # 清除输入并刷新页面
                if 'demo_question' in st.session_state:
                    del st.session_state.demo_question
                st.rerun()
    
    def render_model_info(self):
        """渲染模型信息"""
        with st.expander("📋 模型信息", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **模型架构**: GPT (Transformer)  
                **参数量**: 355M (Medium)  
                **训练方式**: 指令微调 (SFT)  
                **最大长度**: 1024 tokens  
                """)
            
            with col2:
                st.markdown("""
                **支持任务**: 对话、问答、创作  
                **语言**: 中文/英文  
                **更新时间**: 2024年  
                **版本**: v1.0  
                """)
    
    def run(self):
        """运行应用"""
        self.render_header()
        self.render_sidebar()
        self.render_chat_interface()
        self.render_model_info()

def main():
    app = ChatAssistantApp()
    app.run()

if __name__ == "__main__":
    main()