# 归档 AliyunQA_RAG_demo 与 PDF_RAG_demo 的收尾同步

## 现状（已完成，无需重复操作）

- 两目录已 `git mv` 到 `archive/` 并暂存（git status 显示全部 R 状态）
- `data/aliyun_ecs_qa.json` 已复制，与归档目录内的原 `qa.json` md5 一致（副本保留策略符合已确认决定）
- 课堂实验 7 的索引缓存 `rag_index.npz`/`rag_index_meta.json` 不受影响（缓存校验不依赖语料路径）

## 代码修正

- `mini_rag_demo.py`：
  - L34 `CORPUS_PATH` 改为 `os.path.join(SCRIPT_DIR, "data", "aliyun_ecs_qa.json")`
  - L9 docstring 语料说明同步改为 `data/aliyun_ecs_qa.json`

## 依赖清理（requirements.txt）

- 移除 `streamlit>=1.35`、`redis>=5.0`、`tiktoken`、`dashscope>=1.20` 四项（核实过：主线已无使用者，仅归档代码与已归档脚本引用）
- 注释块 `# RAG 相关（AliyunQA_RAG_demo）` 改为指向 mini_rag 的说明，保留 `numpy>=1.26`（mini_rag_demo.py 仍依赖）

## 文档同步（7 处）

1. **README.md**：
   - RAG 表格（L48-49）删除两行，保留 mini_rag 与 wiki_kb
   - 学习路线第 3 条（L67）改为 `mini_rag_demo.py`（无依赖全链路）→ `wiki_kb_agent_demo`（导航式知识库）
2. **AGENTS.md**：删除 L45 `streamlit run AliyunQA_RAG_demo/chat.py` 命令及其注释行
3. **THIRD_PARTY_NOTICES.md**：语料路径更新为 `archive/PDF_RAG_demo/ICBC_2024_FYR.pdf` 与 `archive/AliyunQA_RAG_demo/scrapy-prj-aliyunecs/qa.json`（版权声明随文件移动）
4. **archive/README.md**：新增两条归档登记——
   - `AliyunQA_RAG_demo/`：Scrapy + Redis 向量库 + 旧版模型接入方式已过时，未进入实验手册课堂教学；语料 qa.json 已复制至 `data/aliyun_ecs_qa.json` 供 `mini_rag_demo.py` 使用（目录内保留副本自包含）
   - `PDF_RAG_demo/`：notebook 实现方式陈旧且未进课堂教学；多模态知识库构建另有独立演示覆盖（今后可能单独引入），ICBC 年报语料随目录归档
   - 同时修正既有 `pdf_rag_demo_corpus/` 条目中"`PDF_RAG_demo/` 目录内的原始 ICBC 年报仍为现役语料"的过时表述（现已归档）
5. **wiki_kb_agent_demo/README.md** L73：参考指向改为主线现役实现 `mini_rag_demo.py` 的 `retrieve()`（query 向量化 + 余弦相似度 Top-K），表述改为“可参考 `mini_rag_demo.py` 的检索实现”
6. **实验手册.md**：已核实全文无旧路径引用，无需改动
7. **课堂实录与教学要点.md**：已核实无引用，无需改动

## 验证

- `python mini_rag_demo.py --query "ECS 无法连接怎么排查" --top_k 1`：应显示"从缓存加载"且正常回答（语料内容未变，不触发重建）
- 全仓 grep `AliyunQA_RAG_demo|PDF_RAG_demo` 确认残留引用仅存在于 archive/ 与 THIRD_PARTY_NOTICES 的正确路径中
- `git status` 确认无遗漏的未暂存移动残留

## 假设

- 不执行 git commit（如需提交请明确告知）
- `data/` 目录暂不新增说明文件，若希望加 README 请另行说明