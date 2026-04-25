import re
import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

MAX_ROUNDS = 5

# ---- プロンプト ----
CLARIFY_START_PROMPT = """
あなたはプロジェクト計画のファシリテーターです。
ユーザーのざっくりしたゴールに対して、良い計画を作るために必要な確認質問を3〜4個、
番号付きリストで出力してください。

ルール：
- 1行に1つの質問のみ（改行なし）
- 余計な前置き・後書き不要
- 番号は「1.」形式で統一

確認すべき観点（目標に合わせて選ぶ）：
- 期限・スケジュール感
- 予算・リソース（人数・ツールなど）
- 制約条件や前提
- 完了の定義（何ができたら成功か）
- 懸念していること

【ユーザーの目標（ざっくり）】
{rough_goal}
"""

NEXT_ROUND_PROMPT = """
あなたはプロジェクト計画のファシリテーターです。

【プロジェクトゴール】
{rough_goal}

【これまでのQ&A】
{previous_qa}

上記の情報をもとに、計画作成に十分な情報が揃っているか判断してください。

■ 十分な情報が揃っている場合：
1行目に「DONE」とだけ出力してください。それ以外は何も出力しないでください。

■ まだ不足している場合：
追加で確認すべき質問を2〜3個、番号付きリストで出力してください。
ルール：
- すでに聞いた内容と重複しないこと
- 1行に1つの質問のみ（改行なし）
- 余計な前置き・後書き不要
- 番号は「1.」形式で統一
"""

PLAN_PROMPT = """
あなたはプロジェクトマネジメントの専門家です。
以下のプロジェクト情報をもとに、下記3つを**必ずこの順番・見出し名で**作成してください。

## WBS（作業分解構造）
フェーズ・タスク・サブタスクを階層的に列挙してください。
各タスクには担当者（個人プロジェクトなので「自分」でOK）と目安工数（日数または時間）を付記してください。

## スケジュール案
フェーズごとの開始・終了時期を週単位で示してください。
マイルストーンを明示し、テーブル形式で出力してください。

## リスク一覧
想定されるリスクを列挙してください。
各リスクについて「発生可能性（高/中/低）」「影響度（大/中/小）」「対策」を含むテーブル形式で出力してください。

---
【プロジェクト情報】
{goal_context}
"""

PLAN_CHAT_PROMPT = """
以下のプロジェクト計画について、質問や修正依頼に日本語で答えてください。
回答は具体的かつ簡潔にしてください。

【プロジェクト情報】
{goal_context}

【現在の計画】
{plan}

【質問・依頼】
{question}
"""

# ---- ユーティリティ ----
def parse_questions(text: str) -> list[str]:
    questions = []
    for line in text.strip().splitlines():
        line = line.strip()
        cleaned = re.sub(r'^[\d１２３４５６７８９０]+[\.\)．）]\s*', '', line)
        if cleaned:
            questions.append(cleaned)
    return questions if questions else [text.strip()]

def build_previous_qa(all_rounds: list) -> str:
    lines = []
    for i, r in enumerate(all_rounds):
        lines.append(f"--- ラウンド {i + 1} ---")
        for q, a in zip(r["questions"], r["answers"]):
            lines.append(f"Q: {q}")
            lines.append(f"A: {a if a.strip() else '（未回答）'}")
    return "\n".join(lines)

def build_goal_context(rough_goal: str, all_rounds: list) -> str:
    if not all_rounds:
        return rough_goal
    return f"【ざっくり目標】\n{rough_goal}\n\n【詳細確認のQ&A】\n{build_previous_qa(all_rounds)}"

# ---- Gemini 呼び出し ----
def get_first_questions(rough_goal: str) -> list[str]:
    prompt = CLARIFY_START_PROMPT.format(rough_goal=rough_goal[:3000])
    response = model.generate_content(prompt)
    return parse_questions(response.text)

def get_next_step(rough_goal: str, all_rounds: list) -> tuple[bool, list[str]]:
    """
    Returns (done, next_questions)
    done=True → 情報十分、計画生成へ
    done=False → 追加質問あり
    """
    previous_qa = build_previous_qa(all_rounds)
    prompt = NEXT_ROUND_PROMPT.format(
        rough_goal=rough_goal[:2000],
        previous_qa=previous_qa[:6000],
    )
    response = model.generate_content(prompt)
    text = response.text.strip()
    if text.upper().startswith("DONE"):
        return True, []
    return False, parse_questions(text)

def generate_plan(goal_context: str) -> str:
    prompt = PLAN_PROMPT.format(goal_context=goal_context[:8000])
    response = model.generate_content(prompt)
    return response.text

def refine_plan(goal_context: str, plan: str, question: str) -> str:
    prompt = PLAN_CHAT_PROMPT.format(
        goal_context=goal_context[:3000],
        plan=plan[:10000],
        question=question,
    )
    response = model.generate_content(prompt)
    return response.text

def split_plan(plan_text: str) -> dict:
    sections = {"wbs": "", "schedule": "", "risks": ""}
    parts = plan_text.split("## ")
    for part in parts:
        if part.startswith("WBS"):
            sections["wbs"] = "## " + part.rstrip()
        elif part.startswith("スケジュール"):
            sections["schedule"] = "## " + part.rstrip()
        elif part.startswith("リスク"):
            sections["risks"] = "## " + part.rstrip()
    return sections

# ---- UI ----
st.set_page_config(
    page_title="プロジェクト計画アシスタント",
    page_icon="📋",
    layout="wide",
)

st.title("📋 プロジェクト計画アシスタント")

# セッション状態の初期化
defaults = {
    "phase": "clarify",          # "clarify" or "plan"
    "rough_goal": "",
    "all_rounds": [],            # 完了したラウンドのリスト [{questions, answers}, ...]
    "current_questions": [],     # 現在表示中の質問リスト
    "plan_result": "",
    "plan_chat_history": [],
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

DEMO_GOAL = "個人ブログをWordPressからNext.jsに移行したい。記事数は約50本、独自ドメインあり、デザインも刷新したい。"

def reset_all():
    for key, val in defaults.items():
        st.session_state[key] = val

# ---- サイドバー ----
with st.sidebar:
    st.header("操作")

    if st.button("💡 デモゴールを試す", use_container_width=True):
        reset_all()
        st.session_state.rough_goal = DEMO_GOAL
        st.rerun()

    st.divider()

    if st.session_state.phase == "clarify":
        completed = len(st.session_state.all_rounds)
        if completed == 0 and not st.session_state.current_questions:
            st.info("ゴールを入力してください")
        else:
            # ラウンド進捗をドットで表示
            dots = ""
            for i in range(MAX_ROUNDS):
                if i < completed:
                    dots += "● "
                elif i == completed:
                    dots += "◐ "
                else:
                    dots += "○ "
            st.info(f"ゴール整理中\n{dots.strip()}\nステップ {completed + 1} / 最大{MAX_ROUNDS}")
    else:
        st.success("計画を表示中")

    st.divider()

    if st.button("🔄 最初からやり直す", use_container_width=True):
        reset_all()
        st.rerun()

# ==============================
# STEP 1: ゴール整理フェーズ
# ==============================
if st.session_state.phase == "clarify":

    # ---- 1a: ゴール入力（初回）----
    if not st.session_state.current_questions and not st.session_state.all_rounds:
        st.subheader("ゴールを入力する")
        st.caption("ざっくりした目標を入力してください。AIが必要な確認質問を最大5ステップで行います。")

        rough_goal_input = st.text_area(
            "プロジェクトのゴール（ざっくりでOK）",
            value=st.session_state.rough_goal,
            height=120,
            placeholder="例：副業でECショップを始めたい / 社内の勤怠管理をExcelからシステム化したい",
        )

        if st.button("確認質問を生成 →", type="primary", use_container_width=True):
            if rough_goal_input.strip():
                st.session_state.rough_goal = rough_goal_input.strip()
                with st.spinner("AIが確認事項を考えています..."):
                    questions = get_first_questions(st.session_state.rough_goal)
                st.session_state.current_questions = questions
                st.rerun()
            else:
                st.warning("ゴールを入力してください")

    # ---- 1b: 質問フォーム ----
    elif st.session_state.current_questions:
        round_num = len(st.session_state.all_rounds) + 1
        st.subheader(f"ステップ {round_num} — 確認質問")
        st.caption(f"ゴール：**{st.session_state.rough_goal}**")
        st.divider()

        answers = []
        for i, question in enumerate(st.session_state.current_questions):
            st.markdown(f"**{i + 1}. {question}**")
            answer = st.text_input(
                label=f"回答{i + 1}",
                placeholder="回答を入力（わからなければ空欄でOK）",
                label_visibility="collapsed",
                key=f"answer_r{round_num}_{i}",
            )
            answers.append(answer)
            st.write("")

        st.divider()

        if st.button("次へ →", type="primary", use_container_width=True):
            # 現在のラウンドを保存
            st.session_state.all_rounds.append({
                "questions": st.session_state.current_questions,
                "answers": answers,
            })
            st.session_state.current_questions = []

            completed_rounds = len(st.session_state.all_rounds)

            if completed_rounds >= MAX_ROUNDS:
                # 最大ラウンドに達したら計画生成
                goal_context = build_goal_context(
                    st.session_state.rough_goal,
                    st.session_state.all_rounds,
                )
                with st.spinner("AIが計画を作成中...（少々お待ちください）"):
                    st.session_state.plan_result = generate_plan(goal_context)
                st.session_state.phase = "plan"
            else:
                # AIが次のステップを判断
                with st.spinner("AIが情報を整理しています..."):
                    done, next_questions = get_next_step(
                        st.session_state.rough_goal,
                        st.session_state.all_rounds,
                    )

                if done:
                    goal_context = build_goal_context(
                        st.session_state.rough_goal,
                        st.session_state.all_rounds,
                    )
                    with st.spinner("AIが計画を作成中...（少々お待ちください）"):
                        st.session_state.plan_result = generate_plan(goal_context)
                    st.session_state.phase = "plan"
                else:
                    st.session_state.current_questions = next_questions

            st.rerun()

# ==============================
# STEP 2: 計画表示フェーズ
# ==============================
else:
    st.subheader("プロジェクト計画")
    st.caption(f"ゴール：{st.session_state.rough_goal}")

    sections = split_plan(st.session_state.plan_result)
    goal_context = build_goal_context(
        st.session_state.rough_goal,
        st.session_state.all_rounds,
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📦 WBS（タスク分解）",
        "📅 スケジュール案",
        "⚠️ リスク一覧",
        "💬 Q&Aチャット",
    ])

    with tab1:
        if sections["wbs"]:
            st.markdown(sections["wbs"])
        else:
            st.markdown(st.session_state.plan_result)

    with tab2:
        if sections["schedule"]:
            st.markdown(sections["schedule"])
        else:
            st.info("スケジュール情報が見つかりませんでした。Q&Aチャットで「スケジュール案を出して」と依頼してください。")

    with tab3:
        if sections["risks"]:
            st.markdown(sections["risks"])
        else:
            st.info("リスク情報が見つかりませんでした。Q&Aチャットで「リスク一覧を出して」と依頼してください。")

    with tab4:
        st.caption("計画への質問・修正依頼・深掘りができます")

        for message in st.session_state.plan_chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if question := st.chat_input("例：フェーズ1の期間を1週間延ばして / リスクの対策をもっと詳しく"):
            st.session_state.plan_chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("回答を生成中..."):
                    answer = refine_plan(
                        goal_context,
                        st.session_state.plan_result,
                        question,
                    )
                st.markdown(answer)
                st.session_state.plan_chat_history.append(
                    {"role": "assistant", "content": answer}
                )

    st.divider()
    st.download_button(
        label="📥 計画全体をダウンロード（Markdown）",
        data=f"# プロジェクト計画\n\n## ゴール\n{st.session_state.rough_goal}\n\n---\n\n{st.session_state.plan_result}",
        file_name="project_plan.md",
        mime="text/markdown",
        use_container_width=True,
    )
