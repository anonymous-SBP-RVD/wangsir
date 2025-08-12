# 自动检测文件名的部署脚本
# auto_deploy.py

import subprocess
import sys
import os
import threading
import time
import webbrowser
import glob

class AutoDeployApp:
    def __init__(self):
        self.apps = {}
        self.processes = {}
        self.detect_app_files()
    
    def detect_app_files(self):
        """自动检测应用文件"""
        print("🔍 自动检测应用文件...")
        
        # 可能的文件名模式
        spam_patterns = ['spam_classifier*.py', '*spam*.py', '*classifier*.py']
        chat_patterns = ['chat_assistant*.py', '*chat*.py', '*assistant*.py']
        
        spam_file = None
        chat_file = None
        
        # 检测垃圾短信分类应用
        for pattern in spam_patterns:
            files = glob.glob(pattern)
            for file in files:
                if 'spam' in file.lower() or 'classifier' in file.lower():
                    spam_file = file
                    break
            if spam_file:
                break
        
        # 检测聊天助手应用
        for pattern in chat_patterns:
            files = glob.glob(pattern)
            for file in files:
                if 'chat' in file.lower() or 'assistant' in file.lower():
                    chat_file = file
                    break
            if chat_file:
                break
        
        # 设置应用配置
        if spam_file:
            self.apps['spam_classifier'] = {
                'name': '垃圾短信分类系统',
                'file': spam_file,
                'port': 8501,
                'icon': '🛡️'
            }
            print(f"✅ 找到垃圾短信分类应用: {spam_file}")
        else:
            print("⚠️ 未找到垃圾短信分类应用文件")
        
        if chat_file:
            self.apps['chat_assistant'] = {
                'name': 'AI聊天助手',
                'file': chat_file,
                'port': 8502,
                'icon': '🤖'
            }
            print(f"✅ 找到AI聊天助手应用: {chat_file}")
        else:
            print("⚠️ 未找到AI聊天助手应用文件")
        
        if not self.apps:
            print("❌ 未找到任何应用文件")
            print("📋 当前目录文件:")
            for file in os.listdir('.'):
                if file.endswith('.py'):
                    print(f"   📄 {file}")
    
    def check_requirements(self):
        """检查环境依赖"""
        print("\n🔍 检查环境依赖...")
        
        required_packages = [
            'streamlit',
            'torch', 
            'tiktoken',
            'pandas',
            'plotly'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"❌ {package}")
        
        if missing_packages:
            print(f"\n⚠️ 缺少依赖包: {', '.join(missing_packages)}")
            print("请运行以下命令安装:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
        
        print("✅ 所有依赖已满足")
        return True
    
    def check_model_files(self):
        """检查模型文件"""
        print("\n🔍 检查模型文件...")
        
        # 自动检测模型文件
        model_patterns = {
            '*.pth': '模型权重文件',
            'gptMoudel.py': 'GPT模型定义',
            'generateSimple*.py': '文本生成工具'
        }
        
        found_files = []
        
        for pattern, desc in model_patterns.items():
            if '*' in pattern:
                files = glob.glob(pattern)
                if files:
                    for file in files:
                        found_files.append((file, desc))
                        print(f"✅ {desc}: {file}")
            else:
                if os.path.exists(pattern):
                    found_files.append((pattern, desc))
                    print(f"✅ {desc}: {pattern}")
                else:
                    print(f"⚠️ {desc}: {pattern} (未找到)")
        
        return len(found_files) > 0
    
    def deploy_app(self, app_key):
        """部署单个应用"""
        if app_key not in self.apps:
            print(f"❌ 应用不可用: {app_key}")
            return False
        
        app_info = self.apps[app_key]
        
        try:
            print(f"🚀 启动 {app_info['name']}...")
            print(f"📁 文件: {app_info['file']}")
            print(f"🌐 端口: {app_info['port']}")
            
            # 创建启动命令
            cmd = [
                sys.executable, '-m', 'streamlit', 'run',
                app_info['file'],
                '--server.port', str(app_info['port']),
                '--server.headless', 'true',
                '--browser.serverAddress', 'localhost',
                '--logger.level', 'error'
            ]
            
            # 启动应用
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes[app_key] = process
            
            # 等待启动
            print("⏳ 正在启动，请稍候...")
            time.sleep(5)
            
            if process.poll() is None:
                print(f"✅ {app_info['name']} 启动成功!")
                print(f"🌐 访问地址: http://localhost:{app_info['port']}")
                
                # 尝试打开浏览器
                try:
                    webbrowser.open(f"http://localhost:{app_info['port']}")
                    print("🌐 浏览器已自动打开")
                except:
                    print("⚠️ 无法自动打开浏览器，请手动访问上述地址")
                
                return True
            else:
                print(f"❌ {app_info['name']} 启动失败")
                stdout, stderr = process.communicate()
                if stderr:
                    print(f"错误信息: {stderr}")
                return False
                
        except Exception as e:
            print(f"❌ 启动 {app_info['name']} 时出错: {str(e)}")
            return False
    
    def deploy_all_apps(self):
        """部署所有应用"""
        if not self.apps:
            print("❌ 没有找到可部署的应用文件")
            return
        
        print("🚀 部署所有Web应用...")
        
        success_count = 0
        for app_key in self.apps.keys():
            if self.deploy_app(app_key):
                success_count += 1
            time.sleep(3)  # 错开启动时间
        
        if success_count > 0:
            print(f"\n✅ 成功启动 {success_count} 个应用!")
            print("\n📱 访问地址:")
            for app_key, app_info in self.apps.items():
                if app_key in self.processes and self.processes[app_key].poll() is None:
                    print(f"{app_info['icon']} {app_info['name']}: http://localhost:{app_info['port']}")
        else:
            print("❌ 没有应用启动成功")
    
    def check_app_status(self):
        """检查应用运行状态"""
        print("\n📋 应用运行状态:")
        
        if not self.apps:
            print("❌ 没有找到可用的应用")
            return
        
        running_count = 0
        for app_key, app_info in self.apps.items():
            if app_key in self.processes:
                process = self.processes[app_key]
                if process.poll() is None:
                    print(f"✅ {app_info['name']}: 运行中 (PID: {process.pid})")
                    print(f"   🌐 访问地址: http://localhost:{app_info['port']}")
                    running_count += 1
                else:
                    print(f"❌ {app_info['name']}: 已停止")
                    del self.processes[app_key]
            else:
                print(f"⚪ {app_info['name']}: 未启动")
        
        if running_count == 0:
            print("\n💡 提示: 当前没有运行中的应用")
    
    def stop_all_apps(self):
        """停止所有应用"""
        if not self.processes:
            print("💡 当前没有运行中的应用")
            return
        
        print("\n🛑 停止所有应用...")
        
        for app_key, process in list(self.processes.items()):
            try:
                print(f"🛑 正在停止: {self.apps[app_key]['name']}")
                process.terminate()
                
                try:
                    process.wait(timeout=5)
                    print(f"✅ 已停止: {self.apps[app_key]['name']}")
                except subprocess.TimeoutExpired:
                    process.kill()
                    print(f"🔪 强制停止: {self.apps[app_key]['name']}")
                    
            except Exception as e:
                print(f"⚠️ 停止应用时出错: {str(e)}")
        
        self.processes.clear()
        print("✅ 所有应用已停止")
    
    def show_available_apps(self):
        """显示可用应用"""
        print("\n📱 可用应用:")
        if self.apps:
            for i, (app_key, app_info) in enumerate(self.apps.items(), 1):
                print(f"{i}. {app_info['icon']} {app_info['name']} ({app_info['file']})")
        else:
            print("❌ 未找到任何应用文件")
            print("\n💡 请确保以下文件存在:")
            print("   📄 spam_classifier_app.py (垃圾短信分类)")
            print("   📄 chat_assistant_app.py (AI聊天助手)")
    
    def show_menu(self):
        """显示主菜单"""
        print("\n" + "="*55)
        print("🚀 大模型Web应用自动部署工具")
        print("="*55)
        
        if not self.apps:
            print("❌ 未找到应用文件，请检查文件名")
            print("0. 🚪  退出")
        else:
            menu_items = []
            for i, (app_key, app_info) in enumerate(self.apps.items(), 1):
                print(f"{i}. {app_info['icon']} 启动{app_info['name']}")
                menu_items.append(app_key)
            
            next_num = len(self.apps) + 1
            print(f"{next_num}. 🌐  启动所有应用")
            print(f"{next_num + 1}. 📋  查看应用状态")
            print(f"{next_num + 2}. 🛑  停止所有应用")
            print("0. 🚪  退出")
        
        print("="*55)
    
    def run(self):
        """运行部署工具"""
        print("🎉 欢迎使用大模型Web应用自动部署工具!")
        
        # 显示检测到的文件
        self.show_available_apps()
        
        # 环境检查
        if not self.check_requirements():
            print("\n❌ 环境检查失败，请先安装所需依赖")
            return
        
        self.check_model_files()
        
        if not self.apps:
            print("\n❌ 没有找到可部署的应用文件，程序退出")
            return
        
        try:
            while True:
                self.show_menu()
                choice = input(f"\n请选择操作 (0-{len(self.apps) + 3}): ").strip()
                
                if choice == '0':
                    self.stop_all_apps()
                    print("👋 感谢使用，再见!")
                    break
                elif choice.isdigit():
                    choice_num = int(choice)
                    app_keys = list(self.apps.keys())
                    
                    if 1 <= choice_num <= len(app_keys):
                        # 启动单个应用
                        app_key = app_keys[choice_num - 1]
                        self.deploy_app(app_key)
                    elif choice_num == len(app_keys) + 1:
                        # 启动所有应用
                        self.deploy_all_apps()
                    elif choice_num == len(app_keys) + 2:
                        # 查看状态
                        self.check_app_status()
                    elif choice_num == len(app_keys) + 3:
                        # 停止所有应用
                        self.stop_all_apps()
                    else:
                        print("❌ 无效选择")
                else:
                    print("❌ 请输入有效的数字")
                
                input("\n按Enter键继续...")
        
        except KeyboardInterrupt:
            print("\n\n🛑 收到中断信号，正在停止所有应用...")
            self.stop_all_apps()
            print("👋 再见!")

def main():
    """主函数"""
    deployer = AutoDeployApp()
    deployer.run()

if __name__ == "__main__":
    main()