import streamlit as st
from markitdown import MarkItDown
import requests
import io
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import openai
from typing import TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage
import json
import os
from langsmith import Client
import uuid
from datetime import datetime

class ChatState(TypedDict):
    messages: List[BaseMessage]
    support_analysis: str

st.set_page_config(page_title="个人简历写作助手", layout="wide")

# 读取API KEY
api_key = st.secrets.get("OPENROUTER_API_KEY", "")

# 读取LangSmith配置（从secrets或环境变量）
langsmith_api_key = st.secrets.get("LANGSMITH_API_KEY", os.environ.get("LANGSMITH_API_KEY", ""))
langsmith_project = st.secrets.get("LANGSMITH_PROJECT", os.environ.get("LANGSMITH_PROJECT", "cv-assistant"))

# 设置LangSmith环境变量
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = langsmith_project

# 初始化LangSmith客户端（如果配置了API Key）
langsmith_client = None
if langsmith_api_key:
    try:
        langsmith_client = Client(api_key=langsmith_api_key)
        # 创建项目（如果不存在）
        projects = langsmith_client.list_projects()
        project_names = [p.name for p in projects]
        if langsmith_project not in project_names:
            langsmith_client.create_project(langsmith_project)
    except Exception as e:
        st.warning(f"LangSmith初始化失败: {str(e)}")

# 初始化session state用于存储提示词和文件内容
if "persona" not in st.session_state:
    st.session_state.persona = """请作为专业简历顾问，帮助用户整理个人简历中的"经历"部分。您将基于用户上传的结构化素材表和文档分析专家提供的辅助文档分析报告，生成针对不同类型经历的高质量简历要点。您擅长提炼关键信息，打造以能力和成果为导向的简历内容。
"""
if "task" not in st.session_state:
    st.session_state.task = """经历分类处理：
● 科研项目经历：包括课题研究、毕业论文、小组科研项目等学术性质的活动
● 实习工作经历：包括正式工作、实习、兼职等与就业相关的经历
● 课外活动经历：包括学生会、社团、志愿者、比赛等非学术非就业的活动经历（注意：奖项不属于经历，不要将获奖信息作为经历处理）

奖项单独整理：
● 奖项荣誉：包括各类竞赛奖项、学术荣誉、奖学金等表彰（这些内容将在"奖项列表"部分单独呈现，不要混入经历部分）

工作流程：
1. 首先仔细阅读用户上传的结构化素材表，完整提取所有经历信息，明确区分经历与奖项
2. 然后详细阅读文档分析专家提供的辅助文档分析报告，这是对用户上传的辅助文档(如项目报告、作品集等)的分析结果
3. 交叉比对和整合这两种信息源：
●若文档分析报告中有素材表中未提及的经历，将其添加到相应类别
●若文档分析报告中有素材表经历的补充信息，结合这些信息丰富经历描述
●若两种来源对同一经历有不同描述，优先采用更详细、更具体的描述
4. 确保每个经历都整合了所有可用信息，使描述尽可能完整
5. 按照下方格式整理每类经历，并严格按时间倒序排列（以项目开始时间为准，开始时间越晚的排在越前面）；对于缺乏明确开始时间信息的经历，放置在该分类的最后

重要提示：
1. 必须完整展示结构化素材表中的所有经历条目，结构化素材表中即使项目名称、公司名称或其他关键信息高度重合的条目，也要视为不同经历，全部单独展示
2. 当辅助文档分析报告中的经历与结构化素材表中的经历重合时，应合并这些信息以丰富经历描述
3. 严格区分经历与奖项，不要将奖项信息编入经历描述中
4. 奖项和证书必须直接使用素材表中的原始名称和描述，不要加工或修改
5. 即使某些经历在文档分析报告中未提及，也必须从素材表中提取并整理

要点创作重点指南，针对不同类型经历，要点应有不同侧重：
1. 科研项目经历要点重点突出：
● 专业知识应用（如算法、模型、理论等）
● 研究方法/技能（如数据分析、实验设计、文献综述等）
● 量化成果（如系统性能提升、实验效果改进、发表成果等）
● 迁移能力（如批判性思维、跨学科合作能力等）

2. 实习工作经历要点重点突出：
● 专业技能应用（如编程语言、设计工具、分析软件等）
● 问题解决（如业务挑战、技术难题、流程优化等）
● 量化成果（如效率提升、成本节约、用户增长等）
● 职场能力（如项目管理、团队协作、跨部门沟通等）

3. 课外活动经历要点重点突出：
● 领导/协作能力（如团队管理、成员协调等）
● 沟通/组织能力（如活动策划、资源整合等）
● 创新/解决问题（如创意执行、危机处理等）
● 成果/影响（如活动规模、参与人数、社会影响等）

信息整合检查项：
● 确认已从素材表中提取所有经历，素材表中的每一条经历都必须完整展示，不遗漏
● 确认当支持文件分析报告中有与素材表经历重合的信息时，已将其合并整合
● 确认每个经历都有完整的基本信息（时间、组织、角色）
● 确认所有量化成果和具体技能都已整合
● 确认描述中没有冗余或矛盾的信息
● 确认信息缺失提示针对性强，有实际参考价值
● 确认没有将奖项错误地归类为经历
● 确认每类经历都按开始时间倒序排列（开始时间越晚排越前）
● 确认奖项和证书使用了素材表中的原始名称，没有进行修改或加工
"""
if "output_format" not in st.session_state:
    st.session_state.output_format = """个人简历经历整理报告

【科研经历】
经历一：[项目名称]
时间段：[时间]（以开始时间-结束时间格式呈现）
组织：[组织名称]
角色：[岗位/角色]（如有缺失用[预测]标记）
核心要点：
[专业知识应用] 具体内容
[研究方法/技能] 具体内容
[量化成果] 具体内容
[迁移能力] 具体内容
信息缺失提示：如有关键信息缺失，在此提供补充建议，如"缺少项目具体成果，建议补充量化数据"

经历二：[项目名称]
时间段：[时间]（以开始时间-结束时间格式呈现）
组织：[组织名称]
角色：[岗位/角色]（如有缺失用[预测]标记）
核心要点：
[专业知识应用] 具体内容
[研究方法/技能] 具体内容
[量化成果] 具体内容
[迁移能力] 具体内容
信息缺失提示：如有关键信息缺失，在此提供补充建议

【实习工作经历】
经历一：
时间段：[时间]（以开始时间-结束时间格式呈现）
组织：[组织名称]
角色：[岗位/角色]（如有缺失用[预测]标记）
核心要点：
[专业技能应用] 具体内容
[问题解决] 具体内容
[量化成果] 具体内容
[职场能力] 具体内容
信息缺失提示：如有关键信息缺失，在此提供补充建议

【课外活动经历】
经历一：
时间段：[时间]（以开始时间-结束时间格式呈现）
组织：[组织名称]
角色：[岗位/角色]（如有缺失用[预测]标记）
核心要点：
[领导/协作能力] 具体内容
[沟通/组织能力] 具体内容
[创新/解决问题] 具体内容
[成果/影响] 具体内容
信息缺失提示：如有关键信息缺失，在此提供补充建议

【奖项列表】
奖项一：[奖项名称，直接使用素材表中的原始名称]
获奖时间：[时间]
颁发机构：[组织名称]
奖项级别：[国家级/省级/校级等]（如有缺失用[预测]标记）

奖项二：[奖项名称，直接使用素材表中的原始名称]
获奖时间：[时间]
颁发机构：[组织名称]
奖项级别：[国家级/省级/校级等]（如有缺失用[预测]标记）

在报告最后，添加一个总结部分，简要评估整体简历的强项和需要改进的地方，给出2-3条具体建议。
"""
if "resume_content" not in st.session_state:
    st.session_state.resume_content = ""
if "support_files_content" not in st.session_state:
    st.session_state.support_files_content = []
if "report_result" not in st.session_state:
    st.session_state.report_result = ""
    
# 新增：支持文件分析agent的提示词
if "support_analyst_persona" not in st.session_state:
    st.session_state.support_analyst_persona = """您是一位专业的文档分析专家，擅长从各类文件中提取和整合信息。您的任务是分析用户上传的辅助文档（如项目海报、报告或作品集），并生成标准化报告，用于简历顾问后续处理。您具备敏锐的信息捕捉能力和系统化的分析方法，能够从复杂文档中提取关键经历信息。"""
if "support_analyst_task" not in st.session_state:
    st.session_state.support_analyst_task = """经历分类： 
● 科研项目经历：包括课题研究、毕业论文、小组科研项目等学术性质的活动 
● 实习工作经历：包括正式工作、实习、兼职等与就业相关的经历 
● 课外活动经历：包括学生会、社团、志愿者、比赛等非学术非就业的活动经历
文件分析工作流程：
1.仔细阅读所有上传文档，不遗漏任何页面和部分内容
2.识别文档中所有经历条目，无论是否完整
3.将多个文档中的相关信息进行交叉比对和整合
4.为每个经历创建独立条目，保留所有细节
5.按照下方格式整理每类经历，并按时间倒序排列

信息整合策略： 
● 多文件整合：系统性关联所有辅助文档信息，确保全面捕捉关键细节 
● 团队项目处理：对于团队项目，明确关注个人在团队中的具体职责和贡献，避免笼统描述团队成果 
● 内容优先级：专业知识与能力 > 可量化成果 > 职责描述 > 个人感受 
● 表达原则：简洁精准、专业导向、成果突显、能力凸显
"""
if "support_analyst_output_format" not in st.session_state:
    st.session_state.support_analyst_output_format = """辅助文档分析报告
【科研经历】 
经历一： [项目名称]
时间段：[时间] 
组织：[组织名称] 
角色：[岗位/角色]（如有缺失用[未知]标记） 
核心信息：
●[项目描述] 具体内容
●[使用技术/方法] 具体内容
●[个人职责] 具体内容
●[项目成果] 具体内容 信息完整度评估：[完整/部分完整/不完整] 缺失信息：[列出缺失的关键信息]

经历二： [按照相同格式列出]

【实习工作经历】 
经历一： 
时间段：[时间] 
组织：[组织名称] 
角色：[岗位/角色]（如有缺失用[未知]标记） 
核心信息：
●[工作职责] 具体内容
●[使用技术/工具] 具体内容
●[解决问题] 具体内容
●[工作成果] 具体内容 信息完整度评估：[完整/部分完整/不完整] 缺失信息：[列出缺失的关键信息]

经历二： [按照相同格式列出]

【课外活动经历】 
经历一： 
时间段：[时间] 
组织：[组织名称] 
角色：[岗位/角色]（如有缺失用[未知]标记） 
核心信息：
●[活动描述] 具体内容
●[个人职责] 具体内容
●[使用能力] 具体内容
●[活动成果] 具体内容 信息完整度评估：[完整/部分完整/不完整] 缺失信息：[列出缺失的关键信息]

经历二： [按照相同格式列出]
注意：如果用户未上传任何辅助文档，请直接回复："未检测到辅助文档，无法生成文档分析报告。"
"""

# 新增：简历生成agent的提示词
if "resume_generator_persona" not in st.session_state:
    st.session_state.resume_generator_persona = """您是一位专业的简历撰写专家，擅长将经历整理报告转化为精炼、专业的简历文本。您的任务是基于已生成的经历整理报告，创建一份格式规范、突出亮点的正式简历。您具备优秀的文案能力和排版感，能将分析结果转化为雇主喜欢的简历内容。"""
    
if "resume_generator_task" not in st.session_state:
    st.session_state.resume_generator_task = """简历生成任务：
1. 基于经历整理报告，创建一份完整、专业的简历文本
2. 关注重点：
   ● 将所有核心要点转化为简洁有力的简历条目
   ● 每个条目以行动动词开头，突出成果和能力
   ● 剔除报告中的"信息缺失提示"等非正式内容
   ● 保持专业、简洁的表达方式
3. 结构优化：
   ● 保持经历的分类（科研、实习、课外活动）
   ● 按报告中的时间顺序排列各项经历
   ● 为每个条目添加适当的缩进和格式标记
4. 语言提升：
   ● 使用专业术语替换一般性表达
   ● 增强数据和成果的可视性
   ● 确保语言连贯、流畅

简历创作指南：
● 简历长度：控制在1-2页纸内（约600-1200字）
● 语言风格：专业、简洁、成果导向
● 排版原则：清晰的层级结构，一致的格式
● 重点突出：技能、成果、专业能力应当最为醒目
● 避免内容：个人评价、主观描述、冗长解释
"""

if "resume_generator_output_format" not in st.session_state:
    st.session_state.resume_generator_output_format = """# 个人简历

## 教育背景
【从经历整理报告中提取并格式化】

## 专业技能
【从经历整理报告中提取相关技能并整理】

## 科研项目经历
**项目名称**（时间段）
组织：组织名称
角色：角色名称
- 运用XX技术/方法，完成了XX工作，实现了XX成果
- 负责XX任务，采用XX方法，解决了XX问题
- 通过XX手段，提升了XX性能/效率，达到XX目标

**项目名称**（时间段）
组织：组织名称
角色：角色名称
- 简历条目1
- 简历条目2
- 简历条目3

## 实习工作经历
**公司/组织名称**（时间段）
职位：职位名称
- 简历条目1
- 简历条目2
- 简历条目3

## 课外活动经历
**活动/组织名称**（时间段）
角色：角色名称
- 简历条目1
- 简历条目2
- 简历条目3

## 获奖情况
- 奖项名称，颁发机构，获奖时间
- 奖项名称，颁发机构，获奖时间
"""

# 初始化模型选择
if "selected_support_analyst_model" not in st.session_state:
    st.session_state.selected_support_analyst_model = ""
    
if "selected_cv_assistant_model" not in st.session_state:
    st.session_state.selected_cv_assistant_model = ""
    
if "selected_resume_generator_model" not in st.session_state:
    st.session_state.selected_resume_generator_model = ""

# 获取模型列表（从secrets读取，逗号分隔）
def get_model_list():
    model_str = st.secrets.get("OPENROUTER_MODEL", "")
    if model_str:
        return [m.strip() for m in model_str.split(",") if m.strip()]
    else:
        return ["qwen/qwen-max", "deepseek/deepseek-chat-v3-0324:free", "qwen-turbo", "其它模型..."]

# 在get_model_list()函数定义之后初始化模型选择变量
if "selected_support_analyst_model" not in st.session_state:
    st.session_state.selected_support_analyst_model = get_model_list()[0]
if "selected_cv_assistant_model" not in st.session_state:
    st.session_state.selected_cv_assistant_model = get_model_list()[0]
if "selected_resume_generator_model" not in st.session_state:
    st.session_state.selected_resume_generator_model = get_model_list()[0]

# 保存提示词到文件
def save_prompts():
    prompts = {
        "persona": st.session_state.persona,
        "task": st.session_state.task,
        "output_format": st.session_state.output_format,
        # 新增：支持文件分析agent的提示词
        "support_analyst_persona": st.session_state.support_analyst_persona,
        "support_analyst_task": st.session_state.support_analyst_task,
        "support_analyst_output_format": st.session_state.support_analyst_output_format,
        # 新增：简历生成agent的提示词
        "resume_generator_persona": st.session_state.resume_generator_persona,
        "resume_generator_task": st.session_state.resume_generator_task,
        "resume_generator_output_format": st.session_state.resume_generator_output_format
    }
    # 创建保存目录
    os.makedirs("prompts", exist_ok=True)
    with open("prompts/saved_prompts.json", "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    return True

# 加载保存的提示词
def load_prompts():
    # 禁用加载保存的提示词，直接使用代码中定义的提示词
    return False
    # 以下代码被注释掉，不再从文件加载提示词
    """
    try:
        if os.path.exists("prompts/saved_prompts.json"):
            with open("prompts/saved_prompts.json", "r", encoding="utf-8") as f:
                prompts = json.load(f)
                st.session_state.persona = prompts.get("persona", "")
                st.session_state.task = prompts.get("task", "")
                st.session_state.output_format = prompts.get("output_format", "")
                # 新增：支持文件分析agent的提示词
                st.session_state.support_analyst_persona = prompts.get("support_analyst_persona", "")
                st.session_state.support_analyst_task = prompts.get("support_analyst_task", "")
                st.session_state.support_analyst_output_format = prompts.get("support_analyst_output_format", "")
            return True
        return False
    except Exception as e:
        st.error(f"加载提示词失败: {str(e)}")
        return False
    """

# 读取文件内容函数
def read_file(file):
    try:
        file_bytes = file.read()
        file_stream = io.BytesIO(file_bytes)
        md = MarkItDown()
        raw_content = md.convert(file_stream)
        return raw_content
    except Exception as e:
        return f"[MarkItDown 解析失败: {e}]"

# 处理模型调用
def process_with_model(support_analyst_model, cv_assistant_model, resume_content, support_files_content, 
                      persona, task, output_format, 
                      support_analyst_persona, support_analyst_task, support_analyst_output_format):
    
    # 主运行ID
    main_run_id = str(uuid.uuid4())
    
    # 检查是否有支持文件
    has_support_files = len(support_files_content) > 0
    
    # 显示处理进度
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 1. 如果有支持文件，先用支持文件分析agent处理
        support_analysis_result = ""
        if has_support_files:
            status_text.text("第一阶段：正在分析支持文件...")
            
            # 准备支持文件的内容
            support_files_text = ""
            for i, content in enumerate(support_files_content):
                support_files_text += f"--- 文件 {i+1} ---\n{content}\n\n"
            
            # 构建支持文件分析agent的提示词
            support_prompt = f"""人物设定：{support_analyst_persona}

任务描述：{support_analyst_task}

输出格式：{support_analyst_output_format}

支持文件内容：
{support_files_text}
"""
            
            # 调用支持文件分析agent
            support_analysis_result = run_agent(
                "supporting_doc_analyst",
                support_analyst_model,
                support_prompt,
                main_run_id
            )
            
            progress_bar.progress(50)
            status_text.text("第一阶段完成：支持文件分析完毕")
        else:
            progress_bar.progress(50)
            status_text.text("未提供支持文件，跳过第一阶段分析")
        
        # 2. 准备简历助手的输入
        status_text.text("第二阶段：正在生成简历...")
        
        # 准备简历素材内容
        file_contents = f"简历素材内容:\n{resume_content}\n\n"
        
        # 如果有支持文件分析结果，添加到提示中
        if has_support_files and support_analysis_result:
            file_contents += f"支持文件分析结果:\n{support_analysis_result}\n\n"
        
        # 或者直接添加原始支持文件内容（如果没有分析结果但有支持文件）
        elif has_support_files:
            file_contents += "支持文件内容:\n"
            for i, content in enumerate(support_files_content):
                file_contents += f"--- 文件 {i+1} ---\n{content}\n\n"
        
        # 构建最终的简历助手提示词
        cv_prompt = f"""人物设定：{persona}

任务描述：{task}

输出格式：{output_format}

文件内容：
{file_contents}
"""
        
        # 调用简历助手agent
        final_result = run_agent(
            "cv_assistant", 
            cv_assistant_model,
            cv_prompt,
            main_run_id
        )
        
        progress_bar.progress(100)
        status_text.text("处理完成！")
        
        # 显示结果
        st.markdown(final_result, unsafe_allow_html=True)
        
        # 显示LangSmith链接（如果启用）
        if langsmith_api_key:
            langsmith_base_url = st.secrets.get("LANGSMITH_BASE_URL", "https://smith.langchain.com")
            langsmith_url = f"{langsmith_base_url}/projects/{langsmith_project}/runs/{main_run_id}"
            st.info(f"在LangSmith中查看此运行: [打开监控面板]({langsmith_url})")
            
        # 返回结果，以便保存到session_state
        return final_result
            
    except Exception as e:
        progress_bar.progress(100)
        status_text.text("处理出错！")
        st.error(f"处理失败: {str(e)}")
        st.error("详细错误信息：")
        import traceback
        st.code(traceback.format_exc())
        return None

# 生成最终简历的函数
def generate_resume(model, report_result, persona, task, output_format):
    # 显示处理进度
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 创建运行ID
        run_id = str(uuid.uuid4())
        
        status_text.text("正在生成简历...")
        progress_bar.progress(50)
        
        # 构建简历生成提示词
        resume_prompt = f"""人物设定：{persona}

任务描述：{task}

输出格式：{output_format}

经历整理报告：
{report_result}
"""
        
        # 调用简历生成agent
        final_resume = run_agent(
            "resume_generator", 
            model,
            resume_prompt,
            run_id
        )
        
        progress_bar.progress(100)
        status_text.text("简历生成完成！")
        
        # 显示结果
        st.markdown("## 生成的简历")
        st.markdown("---")
        st.markdown(final_resume, unsafe_allow_html=True)
        
        # 显示LangSmith链接（如果启用）
        if langsmith_api_key:
            langsmith_base_url = st.secrets.get("LANGSMITH_BASE_URL", "https://smith.langchain.com")
            langsmith_url = f"{langsmith_base_url}/projects/{langsmith_project}/runs/{run_id}"
            st.info(f"在LangSmith中查看此运行: [打开监控面板]({langsmith_url})")
        
        # 提供下载链接
        st.download_button(
            label="下载简历",
            data=final_resume,
            file_name="我的简历.md",
            mime="text/markdown"
        )
            
    except Exception as e:
        progress_bar.progress(100)
        status_text.text("简历生成出错！")
        st.error(f"生成失败: {str(e)}")
        st.error("详细错误信息：")
        import traceback
        st.code(traceback.format_exc())

# 运行单个Agent的函数
def run_agent(agent_name, model, prompt, parent_run_id=None):
    # 创建agent运行ID
    agent_run_id = str(uuid.uuid4())
    
    # 在LangSmith中记录agent运行开始（如果启用）
    if langsmith_client and parent_run_id:
        try:
            metadata = {
                "model": model,
                "agent": agent_name,
                "timestamp": datetime.now()
            }
            # 记录agent运行
            langsmith_client.create_run(
                name=f"{agent_name}",
                run_type="chain",
                inputs={"prompt": prompt},
                project_name=langsmith_project,
                run_id=agent_run_id,
                parent_run_id=parent_run_id,
                extra=metadata
            )
        except Exception as e:
            st.warning(f"LangSmith运行创建失败: {str(e)}")
    
    # 创建 LLM 实例
    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model=model,
        temperature=0.7,
    )
    
    # 使用 langchain 的 ChatOpenAI 处理信息
    messages = [HumanMessage(content=prompt)]
    result = llm.invoke(messages)
    
    # 在LangSmith中记录LLM调用（如果启用）
    if langsmith_client and parent_run_id:
        try:
            # 更新agent运行结果
            langsmith_client.update_run(
                run_id=agent_run_id,
                outputs={"response": result.content},
                end_time=datetime.now()
            )
        except Exception as e:
            st.warning(f"LangSmith更新运行结果失败: {str(e)}")
    
    return result.content

# Tab布局 - 修改为三个标签页
TAB1, TAB2, TAB3 = st.tabs(["文件上传与分析", "提示词调试", "系统状态"])

with TAB1:
    st.header("上传你的简历素材和支持文件")
    resume_file = st.file_uploader("个人简历素材表（单选）", type=["pdf", "docx", "doc", "png", "jpg", "jpeg"], accept_multiple_files=False)
    support_files = st.file_uploader("支持文件（可多选）", type=["pdf", "docx", "doc", "png", "jpg", "jpeg"], accept_multiple_files=True)
    
    # 添加"开始分析"按钮
    if st.button("开始分析", use_container_width=True):
        if not api_key:
            st.error("请在 Streamlit secrets 中配置 OPENROUTER_API_KEY")
        elif not resume_file:
            st.error("请上传简历素材表")
        else:
            # 从session_state获取模型
            support_analyst_model = st.session_state.get("selected_support_analyst_model", get_model_list()[0])
            cv_assistant_model = st.session_state.get("selected_cv_assistant_model", get_model_list()[0])
            
            # 读取文件内容
            resume_content = read_file(resume_file)
            st.session_state.resume_content = resume_content
            
            support_files_content = []
            if support_files:
                for f in support_files:
                    content = read_file(f)
                    support_files_content.append(content)
            st.session_state.support_files_content = support_files_content
            
            # 处理并显示结果
            result = process_with_model(
                support_analyst_model,
                cv_assistant_model,
                st.session_state.resume_content, 
                st.session_state.support_files_content,
                st.session_state.persona,
                st.session_state.task,
                st.session_state.output_format,
                st.session_state.support_analyst_persona,
                st.session_state.support_analyst_task,
                st.session_state.support_analyst_output_format
            )
            
            # 保存结果到会话状态，用于后续生成简历
            if result:
                st.session_state.report_result = result
    
    # 检查是否有报告结果，如果有，显示"生成简历"按钮
    if "report_result" in st.session_state and st.session_state.report_result:
        if st.button("生成简历", key="generate_resume_button", use_container_width=True):
            # 从session_state获取模型
            resume_generator_model = st.session_state.get("selected_resume_generator_model", get_model_list()[0])
            
            # 调用简历生成功能
            generate_resume(
                resume_generator_model,
                st.session_state.report_result,
                st.session_state.resume_generator_persona,
                st.session_state.resume_generator_task,
                st.session_state.resume_generator_output_format
            )

# 添加提示词调试到第二个标签页
with TAB2:
    st.title("提示词调试")
    
    # 创建三个分栏，每个分栏对应一个agent的提示词
    agent1, agent2, agent3 = st.tabs(["辅助文档分析Agent", "简历顾问Agent", "简历生成Agent"])
    
    with agent1:
        st.subheader("辅助文档分析Agent提示词调试")
        
        # 选择模型
        model_list = get_model_list()
        selected_model = st.selectbox(
            "选择模型", 
            model_list,
            index=model_list.index(st.session_state.selected_support_analyst_model) if st.session_state.selected_support_analyst_model in model_list else 0,
            key="support_analyst_model_selector"
        )
        st.session_state.selected_support_analyst_model = selected_model
        
        # 人物设定
        st.text_area(
            "人物设定", 
            value=st.session_state.support_analyst_persona, 
            height=150,
            key="support_analyst_persona_editor"
        )
        
        # 任务描述
        st.text_area(
            "任务描述", 
            value=st.session_state.support_analyst_task, 
            height=300,
            key="support_analyst_task_editor"
        )
        
        # 输出格式
        st.text_area(
            "输出格式", 
            value=st.session_state.support_analyst_output_format, 
            height=300,
            key="support_analyst_output_format_editor"
        )
        
        # 保存按钮
        if st.button("保存辅助文档分析Agent设置", key="save_support_analyst"):
            st.session_state.support_analyst_persona = st.session_state.support_analyst_persona_editor
            st.session_state.support_analyst_task = st.session_state.support_analyst_task_editor
            st.session_state.support_analyst_output_format = st.session_state.support_analyst_output_format_editor
            save_prompts()
            st.success("辅助文档分析Agent设置已保存")
    
    with agent2:
        st.subheader("简历顾问Agent提示词调试")
        
        # 选择模型
        model_list = get_model_list()
        selected_model = st.selectbox(
            "选择模型", 
            model_list,
            index=model_list.index(st.session_state.selected_cv_assistant_model) if st.session_state.selected_cv_assistant_model in model_list else 0,
            key="cv_assistant_model_selector"
        )
        st.session_state.selected_cv_assistant_model = selected_model
        
        # 人物设定
        st.text_area(
            "人物设定", 
            value=st.session_state.persona, 
            height=150,
            key="persona_editor"
        )
        
        # 任务描述
        st.text_area(
            "任务描述", 
            value=st.session_state.task, 
            height=300,
            key="task_editor"
        )
        
        # 输出格式
        st.text_area(
            "输出格式", 
            value=st.session_state.output_format, 
            height=300,
            key="output_format_editor"
        )
        
        # 保存按钮
        if st.button("保存简历顾问Agent设置", key="save_cv_assistant"):
            st.session_state.persona = st.session_state.persona_editor
            st.session_state.task = st.session_state.task_editor
            st.session_state.output_format = st.session_state.output_format_editor
            save_prompts()
            st.success("简历顾问Agent设置已保存")
    
    with agent3:
        st.subheader("简历生成Agent提示词调试")
        
        # 选择模型
        model_list = get_model_list()
        selected_model = st.selectbox(
            "选择模型", 
            model_list,
            index=model_list.index(st.session_state.selected_resume_generator_model) if st.session_state.selected_resume_generator_model in model_list else 0,
            key="resume_generator_model_selector"
        )
        st.session_state.selected_resume_generator_model = selected_model
        
        # 人物设定
        st.text_area(
            "人物设定", 
            value=st.session_state.resume_generator_persona, 
            height=150,
            key="resume_generator_persona_editor"
        )
        
        # 任务描述
        st.text_area(
            "任务描述", 
            value=st.session_state.resume_generator_task, 
            height=300,
            key="resume_generator_task_editor"
        )
        
        # 输出格式
        st.text_area(
            "输出格式", 
            value=st.session_state.resume_generator_output_format, 
            height=300,
            key="resume_generator_output_format_editor"
        )
        
        # 保存按钮
        if st.button("保存简历生成Agent设置", key="save_resume_generator"):
            st.session_state.resume_generator_persona = st.session_state.resume_generator_persona_editor
            st.session_state.resume_generator_task = st.session_state.resume_generator_task_editor
            st.session_state.resume_generator_output_format = st.session_state.resume_generator_output_format_editor
            save_prompts()
            st.success("简历生成Agent设置已保存")

# 添加系统状态到第三个标签页
with TAB3:
    st.title("系统状态")
    
    # 添加重置提示词按钮
    if st.button("重置提示词（使用最新代码中的提示词）", type="primary"):
        # 强制重置提示词为代码中的默认值
        st.session_state.persona = """请作为专业简历顾问，帮助用户整理个人简历中的"经历"部分。您将基于用户上传的结构化素材表和文档分析专家提供的辅助文档分析报告，生成针对不同类型经历的高质量简历要点。您擅长提炼关键信息，打造以能力和成果为导向的简历内容。
"""
        st.session_state.task = """经历分类处理：
● 科研项目经历：包括课题研究、毕业论文、小组科研项目等学术性质的活动
● 实习工作经历：包括正式工作、实习、兼职等与就业相关的经历
● 课外活动经历：包括学生会、社团、志愿者、比赛等非学术非就业的活动经历（注意：奖项不属于经历，不要将获奖信息作为经历处理）

奖项单独整理：
● 奖项荣誉：包括各类竞赛奖项、学术荣誉、奖学金等表彰（这些内容将在"奖项列表"部分单独呈现，不要混入经历部分）

工作流程：
1. 首先仔细阅读用户上传的结构化素材表，完整提取所有经历信息，明确区分经历与奖项
2. 然后详细阅读文档分析专家提供的辅助文档分析报告，这是对用户上传的辅助文档(如项目报告、作品集等)的分析结果
3. 交叉比对和整合这两种信息源：
●若文档分析报告中有素材表中未提及的经历，将其添加到相应类别
●若文档分析报告中有素材表经历的补充信息，结合这些信息丰富经历描述
●若两种来源对同一经历有不同描述，优先采用更详细、更具体的描述
4. 确保每个经历都整合了所有可用信息，使描述尽可能完整
5. 按照下方格式整理每类经历，并严格按时间倒序排列（以项目开始时间为准，开始时间越晚的排在越前面）；对于缺乏明确开始时间信息的经历，放置在该分类的最后

重要提示：
1. 必须完整展示结构化素材表中的所有经历条目，结构化素材表中即使项目名称、公司名称或其他关键信息高度重合的条目，也要视为不同经历，全部单独展示
2. 当辅助文档分析报告中的经历与结构化素材表中的经历重合时，应合并这些信息以丰富经历描述
3. 严格区分经历与奖项，不要将奖项信息编入经历描述中
4. 奖项和证书必须直接使用素材表中的原始名称和描述，不要加工或修改
5. 即使某些经历在文档分析报告中未提及，也必须从素材表中提取并整理

要点创作重点指南，针对不同类型经历，要点应有不同侧重：
1. 科研项目经历要点重点突出：
● 专业知识应用（如算法、模型、理论等）
● 研究方法/技能（如数据分析、实验设计、文献综述等）
● 量化成果（如系统性能提升、实验效果改进、发表成果等）
● 迁移能力（如批判性思维、跨学科合作能力等）

2. 实习工作经历要点重点突出：
● 专业技能应用（如编程语言、设计工具、分析软件等）
● 问题解决（如业务挑战、技术难题、流程优化等）
● 量化成果（如效率提升、成本节约、用户增长等）
● 职场能力（如项目管理、团队协作、跨部门沟通等）

3. 课外活动经历要点重点突出：
● 领导/协作能力（如团队管理、成员协调等）
● 沟通/组织能力（如活动策划、资源整合等）
● 创新/解决问题（如创意执行、危机处理等）
● 成果/影响（如活动规模、参与人数、社会影响等）

信息整合检查项：
● 确认已从素材表中提取所有经历，素材表中的每一条经历都必须完整展示，不遗漏
● 确认当支持文件分析报告中有与素材表经历重合的信息时，已将其合并整合
● 确认每个经历都有完整的基本信息（时间、组织、角色）
● 确认所有量化成果和具体技能都已整合
● 确认描述中没有冗余或矛盾的信息
● 确认信息缺失提示针对性强，有实际参考价值
● 确认没有将奖项错误地归类为经历
● 确认每类经历都按开始时间倒序排列（开始时间越晚排越前）
● 确认奖项和证书使用了素材表中的原始名称，没有进行修改或加工
"""
        st.session_state.output_format = """个人简历经历整理报告

【科研经历】
经历一：[项目名称]
时间段：[时间]（以开始时间-结束时间格式呈现）
组织：[组织名称]
角色：[岗位/角色]（如有缺失用[预测]标记）
核心要点：
[专业知识应用] 具体内容
[研究方法/技能] 具体内容
[量化成果] 具体内容
[迁移能力] 具体内容
信息缺失提示：如有关键信息缺失，在此提供补充建议，如"缺少项目具体成果，建议补充量化数据"

经历二：[项目名称]
时间段：[时间]（以开始时间-结束时间格式呈现）
组织：[组织名称]
角色：[岗位/角色]（如有缺失用[预测]标记）
核心要点：
[专业知识应用] 具体内容
[研究方法/技能] 具体内容
[量化成果] 具体内容
[迁移能力] 具体内容
信息缺失提示：如有关键信息缺失，在此提供补充建议

【实习工作经历】
经历一：
时间段：[时间]（以开始时间-结束时间格式呈现）
组织：[组织名称]
角色：[岗位/角色]（如有缺失用[预测]标记）
核心要点：
[专业技能应用] 具体内容
[问题解决] 具体内容
[量化成果] 具体内容
[职场能力] 具体内容
信息缺失提示：如有关键信息缺失，在此提供补充建议

【课外活动经历】
经历一：
时间段：[时间]（以开始时间-结束时间格式呈现）
组织：[组织名称]
角色：[岗位/角色]（如有缺失用[预测]标记）
核心要点：
[领导/协作能力] 具体内容
[沟通/组织能力] 具体内容
[创新/解决问题] 具体内容
[成果/影响] 具体内容
信息缺失提示：如有关键信息缺失，在此提供补充建议

【奖项列表】
奖项一：[奖项名称，直接使用素材表中的原始名称]
获奖时间：[时间]
颁发机构：[组织名称]
奖项级别：[国家级/省级/校级等]（如有缺失用[预测]标记）

奖项二：[奖项名称，直接使用素材表中的原始名称]
获奖时间：[时间]
颁发机构：[组织名称]
奖项级别：[国家级/省级/校级等]（如有缺失用[预测]标记）

在报告最后，添加一个总结部分，简要评估整体简历的强项和需要改进的地方，给出2-3条具体建议。
"""
        st.session_state.support_analyst_persona = """您是一位专业的文档分析专家，擅长从各类文件中提取和整合信息。您的任务是分析用户上传的辅助文档（如项目海报、报告或作品集），并生成标准化报告，用于简历顾问后续处理。您具备敏锐的信息捕捉能力和系统化的分析方法，能够从复杂文档中提取关键经历信息。"""
        st.session_state.support_analyst_task = """经历分类： 
● 科研项目经历：包括课题研究、毕业论文、小组科研项目等学术性质的活动 
● 实习工作经历：包括正式工作、实习、兼职等与就业相关的经历 
● 课外活动经历：包括学生会、社团、志愿者、比赛等非学术非就业的活动经历
文件分析工作流程：
1.仔细阅读所有上传文档，不遗漏任何页面和部分内容
2.识别文档中所有经历条目，无论是否完整
3.将多个文档中的相关信息进行交叉比对和整合
4.为每个经历创建独立条目，保留所有细节
5.按照下方格式整理每类经历，并按时间倒序排列

信息整合策略： 
● 多文件整合：系统性关联所有辅助文档信息，确保全面捕捉关键细节 
● 团队项目处理：对于团队项目，明确关注个人在团队中的具体职责和贡献，避免笼统描述团队成果 
● 内容优先级：专业知识与能力 > 可量化成果 > 职责描述 > 个人感受 
● 表达原则：简洁精准、专业导向、成果突显、能力凸显
"""
        st.session_state.support_analyst_output_format = """辅助文档分析报告
【科研经历】 
经历一： [项目名称]
时间段：[时间] 
组织：[组织名称] 
角色：[岗位/角色]（如有缺失用[未知]标记） 
核心信息：
●[项目描述] 具体内容
●[使用技术/方法] 具体内容
●[个人职责] 具体内容
●[项目成果] 具体内容 信息完整度评估：[完整/部分完整/不完整] 缺失信息：[列出缺失的关键信息]

经历二： [按照相同格式列出]

【实习工作经历】 
经历一： 
时间段：[时间] 
组织：[组织名称] 
角色：[岗位/角色]（如有缺失用[未知]标记） 
核心信息：
●[工作职责] 具体内容
●[使用技术/工具] 具体内容
●[解决问题] 具体内容
●[工作成果] 具体内容 信息完整度评估：[完整/部分完整/不完整] 缺失信息：[列出缺失的关键信息]

经历二： [按照相同格式列出]

【课外活动经历】 
经历一： 
时间段：[时间] 
组织：[组织名称] 
角色：[岗位/角色]（如有缺失用[未知]标记） 
核心信息：
●[活动描述] 具体内容
●[个人职责] 具体内容
●[使用能力] 具体内容
●[活动成果] 具体内容 信息完整度评估：[完整/部分完整/不完整] 缺失信息：[列出缺失的关键信息]

经历二： [按照相同格式列出]
注意：如果用户未上传任何辅助文档，请直接回复："未检测到辅助文档，无法生成文档分析报告。"
"""
        st.session_state.resume_generator_persona = """您是一位专业的简历撰写专家，擅长将经历整理报告转化为精炼、专业的简历文本。您的任务是基于已生成的经历整理报告，创建一份格式规范、突出亮点的正式简历。您具备优秀的文案能力和排版感，能将分析结果转化为雇主喜欢的简历内容。"""
        
        st.session_state.resume_generator_task = """简历生成任务：
1. 基于经历整理报告，创建一份完整、专业的简历文本
2. 关注重点：
   ● 将所有核心要点转化为简洁有力的简历条目
   ● 每个条目以行动动词开头，突出成果和能力
   ● 剔除报告中的"信息缺失提示"等非正式内容
   ● 保持专业、简洁的表达方式
3. 结构优化：
   ● 保持经历的分类（科研、实习、课外活动）
   ● 按报告中的时间顺序排列各项经历
   ● 为每个条目添加适当的缩进和格式标记
4. 语言提升：
   ● 使用专业术语替换一般性表达
   ● 增强数据和成果的可视性
   ● 确保语言连贯、流畅

简历创作指南：
● 简历长度：控制在1-2页纸内（约600-1200字）
● 语言风格：专业、简洁、成果导向
● 排版原则：清晰的层级结构，一致的格式
● 重点突出：技能、成果、专业能力应当最为醒目
● 避免内容：个人评价、主观描述、冗长解释
"""

        st.session_state.resume_generator_output_format = """# 个人简历

## 教育背景
【从经历整理报告中提取并格式化】

## 专业技能
【从经历整理报告中提取相关技能并整理】

## 科研项目经历
**项目名称**（时间段）
组织：组织名称
角色：角色名称
- 运用XX技术/方法，完成了XX工作，实现了XX成果
- 负责XX任务，采用XX方法，解决了XX问题
- 通过XX手段，提升了XX性能/效率，达到XX目标

**项目名称**（时间段）
组织：组织名称
角色：角色名称
- 简历条目1
- 简历条目2
- 简历条目3

## 实习工作经历
**公司/组织名称**（时间段）
职位：职位名称
- 简历条目1
- 简历条目2
- 简历条目3

## 课外活动经历
**活动/组织名称**（时间段）
角色：角色名称
- 简历条目1
- 简历条目2
- 简历条目3

## 获奖情况
- 奖项名称，颁发机构，获奖时间
- 奖项名称，颁发机构，获奖时间
"""
        st.success("提示词已重置！使用了最新代码中定义的提示词")
    
    # 显示API连接状态
    st.subheader("API连接")
    if api_key:
        st.success("OpenRouter API已配置")
    else:
        st.error("OpenRouter API未配置")
    
    # 显示LangSmith连接状态
    st.subheader("LangSmith监控")
    if langsmith_client:
        st.success(f"已连接到LangSmith")
        st.info(f"项目: {langsmith_project}")
        # 获取最近的运行记录
        try:
            runs = langsmith_client.list_runs(
                project_name=langsmith_project,
                limit=5
            )
            if runs:
                st.write("最近5次运行:")
                for run in runs:
                    st.write(f"- {run.name} ({run.run_type}) - {run.status}")
        except:
            st.write("无法获取最近运行记录")
    else:
        st.warning("LangSmith未配置")
        st.info("如需启用AI监控，请设置LANGSMITH_API_KEY")
    
    # 显示各Agent使用的模型信息
    st.subheader("Agent使用的模型")
    support_analyst_model = st.session_state.get("selected_support_analyst_model", get_model_list()[0])
    cv_assistant_model = st.session_state.get("selected_cv_assistant_model", get_model_list()[0])
    resume_generator_model = st.session_state.get("selected_resume_generator_model", get_model_list()[0])
    
    st.info(f"辅助文档分析Agent模型: {support_analyst_model}")
    st.info(f"简历顾问Agent模型: {cv_assistant_model}")
    st.info(f"简历生成Agent模型: {resume_generator_model}")
    
    # 显示最近的处理状态
    st.subheader("会话状态")
    resume_loaded = "resume_content" in st.session_state and st.session_state.resume_content
    support_loaded = "support_files_content" in st.session_state and len(st.session_state.support_files_content) > 0
    report_generated = "report_result" in st.session_state and st.session_state.report_result
    
    st.write(f"- 简历素材表: {'已上传' if resume_loaded else '未上传'}")
    st.write(f"- 支持文件: {'已上传 (' + str(len(st.session_state.support_files_content)) + '个文件)' if support_loaded else '未上传'}")
    st.write(f"- 经历整理报告: {'已生成' if report_generated else '未生成'}")
    
    # 清除会话状态按钮
    if st.button("清除处理结果", type="secondary"):
        if "report_result" in st.session_state:
            del st.session_state.report_result
        st.success("处理结果已清除，可以开始新的分析")
