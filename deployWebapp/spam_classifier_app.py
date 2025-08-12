# 垃圾短信分类Web应用
# spam_classifier_app.py

import streamlit as st
import torch
import tiktoken
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
from gptMoudel import GPTModel
import torch.nn.functional as F

class SpamClassifierApp:
    def __init__(self):
        self.setup_page_config()
        self.load_model()
        self.setup_session_state()
    
    def setup_page_config(self):
        """设置页面配置"""
        st.set_page_config(
            page_title="智能垃圾短信检测系统",
            page_icon="🛡️",
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
                "emb_dim": 768,
                "n_heads": 12,
                "n_layers": 12,
                "drop_rate": 0.0,
                "qkv_bias": True
            }
            
            # 加载模型
            self.model = GPTModel(self.config)
            self.model.out_head = torch.nn.Linear(
                in_features=self.config["emb_dim"],
                out_features=2  # 二分类
            )
            
            # 加载权重
            if os.path.exists("review_classifier.pth"):
                self.model.load_state_dict(torch.load("review_classifier.pth", map_location="cpu"))
                self.model_loaded = True
            else:
                self.model_loaded = False
            
            self.model.eval()
            self.tokenizer = tiktoken.get_encoding("gpt2")
            
        except Exception as e:
            st.error(f"模型加载失败: {str(e)}")
            self.model_loaded = False
    
    def setup_session_state(self):
        """初始化会话状态"""
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        if 'stats' not in st.session_state:
            st.session_state.stats = {'spam': 0, 'normal': 0, 'total': 0}
    
    def classify_text(self, text, max_length=120):
        """分类文本"""
        if not self.model_loaded:
            return None, 0.0
        
        try:
            # 文本预处理
            input_ids = self.tokenizer.encode(text)
            input_ids = input_ids[:max_length]  # 截断
            
            # 填充到固定长度
            pad_token_id = 50256
            input_ids += [pad_token_id] * (max_length - len(input_ids))
            
            # 转换为张量
            input_tensor = torch.tensor(input_ids).unsqueeze(0)
            
            # 模型推理
            with torch.no_grad():
                logits = self.model(input_tensor)[:, -1, :]
                probabilities = F.softmax(logits, dim=-1)
                predicted_class = torch.argmax(logits, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
            
            return predicted_class, confidence
            
        except Exception as e:
            st.error(f"分类过程出错: {str(e)}")
            return None, 0.0
    
    def save_prediction(self, text, prediction, confidence):
        """保存预测历史"""
        result = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'text': text[:50] + "..." if len(text) > 50 else text,
            'prediction': '垃圾短信' if prediction == 1 else '正常短信',
            'confidence': f"{confidence:.2%}",
            'risk_level': self.get_risk_level(confidence)
        }
        
        st.session_state.prediction_history.insert(0, result)
        
        # 更新统计
        if prediction == 1:
            st.session_state.stats['spam'] += 1
        else:
            st.session_state.stats['normal'] += 1
        st.session_state.stats['total'] += 1
        
        # 保持历史记录在合理范围内
        if len(st.session_state.prediction_history) > 100:
            st.session_state.prediction_history = st.session_state.prediction_history[:100]
    
    def get_risk_level(self, confidence):
        """根据置信度确定风险等级"""
        if confidence > 0.9:
            return "高"
        elif confidence > 0.7:
            return "中"
        else:
            return "低"
    
    def render_header(self):
        """渲染页面头部"""
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="color: #1f77b4; font-size: 3rem; margin-bottom: 0.5rem;">
                🛡️ 智能垃圾短信检测系统
            </h1>
            <p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
                基于GPT架构的深度学习模型，智能识别垃圾短信
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """渲染侧边栏"""
        with st.sidebar:
            st.markdown("### 📊 系统统计")
            
            if not self.model_loaded:
                st.error("❌ 模型未加载")
                st.info("请确保 review_classifier.pth 文件存在")
                return
            
            st.success("✅ 模型已加载")
            
            # 统计信息
            stats = st.session_state.stats
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("总检测数", stats['total'])
                st.metric("垃圾短信", stats['spam'])
            
            with col2:
                st.metric("正常短信", stats['normal'])
                if stats['total'] > 0:
                    spam_rate = stats['spam'] / stats['total'] * 100
                    st.metric("垃圾率", f"{spam_rate:.1f}%")
            
            # 统计图表
            if stats['total'] > 0:
                fig = px.pie(
                    values=[stats['spam'], stats['normal']], 
                    names=['垃圾短信', '正常短信'],
                    title="检测结果分布",
                    color_discrete_map={'垃圾短信': '#ff4b4b', '正常短信': '#00cc88'}
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # 清除历史按钮
            if st.button("🗑️ 清除历史记录", use_container_width=True):
                st.session_state.prediction_history = []
                st.session_state.stats = {'spam': 0, 'normal': 0, 'total': 0}
                st.rerun()
    
    def render_main_content(self):
        """渲染主要内容"""
        # 输入区域
        st.markdown("### 📱 短信内容检测")
        
        # 预设示例
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 垃圾短信示例", use_container_width=True):
                st.session_state.demo_text = "恭喜您中奖了！请立即点击链接领取888元现金大奖，机会难得，仅限今日！回复TD退订"
        
        with col2:
            if st.button("📝 正常短信示例", use_container_width=True):
                st.session_state.demo_text = "您好，您的快递已到达丰巢快递柜，取件码：1234，请及时取件。"
        
        # 文本输入
        text_input = st.text_area(
            "请输入要检测的短信内容：",
            value=st.session_state.get('demo_text', ''),
            height=120,
            placeholder="在这里粘贴或输入短信内容..."
        )
        
        # 检测按钮
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            detect_button = st.button(
                "🔍 开始检测", 
                use_container_width=True,
                type="primary",
                disabled=not text_input.strip() or not self.model_loaded
            )
        
        # 执行检测
        if detect_button and text_input.strip():
            with st.spinner("🧠 AI正在分析中..."):
                prediction, confidence = self.classify_text(text_input.strip())
                
                if prediction is not None:
                    # 保存预测结果
                    self.save_prediction(text_input.strip(), prediction, confidence)
                    
                    # 显示结果
                    self.show_prediction_result(prediction, confidence)
    
    def show_prediction_result(self, prediction, confidence):
        """显示预测结果"""
        st.markdown("### 🎯 检测结果")
        
        if prediction == 1:  # 垃圾短信
            st.error(f"⚠️ **垃圾短信** (置信度: {confidence:.2%})")
            st.markdown("""
            <div style="background-color: #ffebee; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #f44336;">
                <h4>🚨 安全提醒</h4>
                <ul>
                    <li>请勿点击短信中的可疑链接</li>
                    <li>不要回复或提供个人信息</li>
                    <li>建议直接删除此短信</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:  # 正常短信
            st.success(f"✅ **正常短信** (置信度: {confidence:.2%})")
            st.markdown("""
            <div style="background-color: #e8f5e8; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #4caf50;">
                <h4>✅ 安全提醒</h4>
                <p>该短信内容看起来是正常的，但仍建议您保持警惕，确认发送方身份。</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 置信度可视化
        progress_bar = st.progress(confidence)
        risk_level = self.get_risk_level(confidence)
        
        color_map = {"高": "🔴", "中": "🟡", "低": "🟢"}
        st.write(f"**风险等级**: {color_map[risk_level]} {risk_level}")
    
    def render_history(self):
        """渲染历史记录"""
        if st.session_state.prediction_history:
            st.markdown("### 📋 检测历史")
            
            # 转换为DataFrame显示
            df = pd.DataFrame(st.session_state.prediction_history)
            
            # 自定义样式
            def highlight_spam(val):
                color = '#ffebee' if val == '垃圾短信' else '#e8f5e8'
                return f'background-color: {color}'
            
            styled_df = df.style.applymap(highlight_spam, subset=['prediction'])
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True
            )
    
    def run(self):
        """运行应用"""
        self.render_header()
        self.render_sidebar()
        self.render_main_content()
        self.render_history()

def main():
    app = SpamClassifierApp()
    app.run()

if __name__ == "__main__":
    main()