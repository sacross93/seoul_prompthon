from urllib.parse import urlparse, quote
import aiohttp
from playwright.async_api import async_playwright, Page, BrowserContext
from pathlib import Path
import subprocess
import io
from PIL import Image as PILImage
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableLambda
import re
from langchain_core.messages import AIMessage, HumanMessage, ChatMessage, SystemMessage, FunctionMessage, ToolMessage, BaseMessage
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from typing import List, Union
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_core.runnables import chain as chain_decorator
import base64
import platform
import asyncio
from typing import List, Optional, TypedDict
import os
from getpass import getpass

from dotenv import load_dotenv

# Load environment variables
openaijykey='/home/ncp/workspace/nasw337n1/vscode_web/seoul_prompthon/.env'
load_dotenv(openaijykey)
openai_api_key = os.getenv("OPENAI_API_KEY_SSIMU")
os.environ["OPENAI_API_KEY"] = openai_api_key

# GPT model setting
llm = ChatOpenAI(model="gpt-4o", max_tokens=4096)

# Global variable for new tab URL
global_new_tab_url = None

class BBox(TypedDict):
    x: float
    y: float
    text: str
    type: str
    ariaLabel: str

class Prediction(TypedDict):
    action: str
    args: Optional[List[str]]

class AgentState(TypedDict):
    page: Page
    input: str
    img: str
    bboxes: List[BBox]
    prediction: Prediction
    scratchpad: List[BaseMessage]
    observation: str

async def save_page(state: AgentState, save_as_pdf: bool = False):
    page = state["page"]
    download_path = os.path.join(os.getcwd(), "downloads")
    Path(download_path).mkdir(parents=True, exist_ok=True)
    parsed_url = urlparse(page.url)
    domain = parsed_url.netloc.replace(".", "_")
    path = quote(parsed_url.path.strip("/").replace("/", "_"))

    if save_as_pdf:
        pdf_path = os.path.join(download_path, f"{domain}_{path}.pdf" if path else f"{domain}.pdf")
        await page.pdf(path=pdf_path)
        print(f"Page saved as PDF: {pdf_path}")
    else:
        html_file_name = f"{domain}_{path}.html" if path else f"{domain}.html"
        html_path = os.path.join(download_path, html_file_name)
        content = await page.content()
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Page saved as HTML: {html_path}")

async def download_pdf_from_url(pdf_url: str, download_path: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(pdf_url) as response:
            if response.status == 200:
                pdf_data = await response.read()
                with open(download_path, 'wb') as f:
                    f.write(pdf_data)
                print(f"PDF saved successfully to {download_path}")
            else:
                raise Exception(f"Failed to download PDF. Status code: {response.status}")

async def download_page(state: AgentState):
    page = state["page"]
    download_dir = os.path.join(os.getcwd(), "downloads")
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    try:
        embed_selector = 'embed[type="application/pdf"]'
        embed_element = await page.query_selector(embed_selector)
        if embed_element:
            pdf_url = await embed_element.get_attribute('src')
            if not pdf_url or pdf_url == 'about:blank':
                pdf_url = page.url

            parsed_url = urlparse(pdf_url)
            domain = parsed_url.netloc.replace(".", "_")
            path = quote(parsed_url.path.strip("/").replace("/", "_"))
            pdf_file_name = f"{domain}_{path}.pdf" if path else f"{domain}.pdf"
            await download_pdf_from_url(pdf_url, os.path.join(download_dir, pdf_file_name))
            return f"PDF saved successfully to {download_dir}"
        else:
            await save_page(state, save_as_pdf=False)
            return "Page saved as HTML successfully."
    except Exception as e:
        await page.screenshot(path="error.png")
        return f"Failed to save page: {str(e)}"

async def click(state: AgentState):
    global global_new_tab_url

    page = state["page"]
    click_args = state["prediction"]["args"]

    if click_args is None or len(click_args) != 1:
        return f"Failed to click bounding box labeled as number {click_args}"

    bbox_id = click_args[0]
    try:
        bbox_id = int(bbox_id)
        bbox = state["bboxes"][bbox_id]
    except (ValueError, KeyError):
        return f"Error: no bbox for : {bbox_id}"

    x, y = bbox["x"], bbox["y"]

    new_tab = None

    def handle_new_page(page):
        nonlocal new_tab
        new_tab = page

    page.context.on("page", handle_new_page)

    await page.mouse.click(x, y)
    await page.wait_for_timeout(2000)

    page.context.remove_listener("page", handle_new_page)

    if new_tab:
        await new_tab.wait_for_load_state()
        state["page"] = new_tab
        global_new_tab_url = new_tab
        return f"Clicked {bbox_id}"
    else:
        global_new_tab_url = None
        return f"Clicked {bbox_id}"

async def type_text(state: AgentState):
    page = state["page"]
    type_args = state["prediction"]["args"]
    if type_args is None or len(type_args) != 2:
        return f"Failed to type in element from bounding box labeled as number {type_args}"

    bbox_id = type_args[0]
    bbox_id = int(bbox_id)
    bbox = state["bboxes"][bbox_id]
    x, y = bbox["x"], bbox["y"]
    text_content = type_args[1]
    await page.mouse.click(x, y)
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyboard.press(select_all)
    await page.keyboard.press("Backspace")
    await page.keyboard.type(text_content)
    await page.keyboard.press("Enter")
    return f"Typed {text_content} and submitted"

async def scroll(state: AgentState):
    page = state["page"]
    scroll_args = state["prediction"]["args"]
    if scroll_args is None or len(scroll_args) != 2:
        return "Failed to scroll due to incorrect arguments."

    target, direction = scroll_args

    if target.upper() == "WINDOW":
        await page.wait_for_load_state('load')
        viewport_size = page.viewport_size
        if viewport_size:
            center_x = viewport_size['width'] / 2
            center_y = viewport_size['height'] / 2
            await page.mouse.move(center_x, center_y)
            scroll_amount = 500 if direction.lower() == "down" else -500
            await page.mouse.wheel(0, scroll_amount)
            return f"Scrolled {direction} in window"
    else:
        scroll_amount = 200
        target_id = int(target)
        bbox = state["bboxes"][target_id]
        x, y = bbox["x"], bbox["y"]
        scroll_direction = -scroll_amount if direction.lower() == "up" else scroll_amount
        await page.mouse.move(x, y)
        await page.mouse.wheel(0, scroll_direction)

    return f"Scrolled {direction} in {'window' if target.upper() == 'WINDOW' else 'element'}"

async def wait(state: AgentState):
    sleep_time = 5
    await asyncio.sleep(sleep_time)
    return f"Waited for {sleep_time}s."

async def go_back(state: AgentState):
    page = state["page"]
    await page.go_back()
    return f"Navigated back a page to {page.url}."

async def to_google(state: AgentState):
    page = state["page"]
    await page.goto("https://www.google.com/")
    return "Navigated to google.com."

with open("/home/ncp/workspace/nasw337n1/vscode_web/seoul_prompthon/mark_page.js") as f:
    mark_page_script = f.read()

@chain_decorator
async def mark_page(page):
    await page.evaluate(mark_page_script)
    for _ in range(10):
        try:
            bboxes = await page.evaluate("markPage()")
            break
        except:
            asyncio.sleep(3)
    screenshot = await page.screenshot()
    await page.evaluate("unmarkPage()")
    return {
        "img": base64.b64encode(screenshot).decode(),
        "bboxes": bboxes,
    }

async def annotate(state: AgentState):
    marked_page = await mark_page.with_retry().ainvoke(state["page"])
    return {**state, **marked_page}

def format_descriptions(state: AgentState):
    labels = []
    for i, bbox in enumerate(state["bboxes"]):
        text = bbox.get("ariaLabel") or ""
        if not text.strip():
            text = bbox["text"]
        el_type = bbox.get("type")
        labels.append(f'{i} (<{el_type}/>): "{text}"')
    bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)
    return {**state, "bbox_descriptions": bbox_descriptions}

def parse(text: str) -> dict:
    print(text)
    action_prefix = "Action: "
    if not text.strip().split("\n")[-1].startswith(action_prefix):
        return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
    action_block = text.strip().split("\n")[-1]

    action_str = action_block[len(action_prefix):]
    split_output = action_str.split(" ", 1)
    if len(split_output) == 1:
        action, action_input = split_output[0], None
    else:
        action, action_input = split_output
    action = action.strip()
    if action_input is not None:
        action_input = [
            inp.strip().strip("[]") for inp in action_input.strip().split(";")
        ]
    return {"action": action, "args": action_input}

prompt = ChatPromptTemplate(
    input_variables=['bbox_descriptions', 'img', 'input'],
    input_types={'scratchpad': List[Union[AIMessage, HumanMessage,
                                          ChatMessage, SystemMessage, FunctionMessage, ToolMessage]]},
    partial_variables={'scratchpad': []},
    messages=[SystemMessagePromptTemplate(
        prompt=[PromptTemplate(input_variables=[],
                               template="""Imagine you are a robot browsing the web, just like humans. 
                                   Now you need to complete a task. 
                                   In each iteration, you will receive an Observation that includes a screenshot of a webpage and some texts. 
                                   This screenshot will feature Numerical Labels placed in the TOP LEFT corner of each Web Element. 
                                   Carefully analyze the visual information to identify the Numerical Label corresponding to the Web Element that requires interaction, 
                                   then follow the guidelines and choose one of the following actions:\n\n
                                   1. Click a Web Element.\n
                                   2. Delete existing content in a textbox and then type content.\n
                                   3. Scroll up or down.\n
                                   4. Wait \n
                                   5. Go back\n
                                   7. Return to google to start over.\n
                                   8. Download viewing page as file (If a PDF file is open, use Download) \n
                                   9. Respond with the final answer\n\n
                                   Correspondingly, Action should STRICTLY follow the format:\n\n
                                   
                                   - Click [Numerical_Label] \n
                                   - Type [Numerical_Label]; [Content] \n
                                   - Scroll [Numerical_Label or WINDOW]; [up or down] \n
                                   - Wait \n
                                   - GoBack\n
                                   - Google\n
                                   - Download\n
                                   - ANSWER; [content]\n\n
                                   Key Guidelines You MUST follow:\n\n
                                   
                                   * Action guidelines *\n
                                   1) Execute only one action per iteration.\n
                                   2) When clicking or typing, ensure to select the correct bounding box.\n
                                   3) Numeric labels lie in the top-left corner of their corresponding bounding boxes and are colored the same.\n\n
                                   
                                   * Web Browsing Guidelines *\n
                                   1) Don't interact with useless web elements like Login, Sign-in, donation that appear in Webpages\n
                                   2) Select strategically to minimize time wasted.\n\n
                                   
                                   Your reply should strictly follow the format:\n\n
                                   Thought: {{Your brief thoughts (briefly summarize the info that will help ANSWER)}}\n
                                   Action: {{One Action format you choose}}\nThen the User will provide:\n
                                   Observation: {{A labeled screenshot Given by User}}\n""")]),
              MessagesPlaceholder(variable_name='scratchpad', optional=True),
              HumanMessagePromptTemplate(prompt=[ImagePromptTemplate(input_variables=['img'], template={'url': 'data:image/png;base64,{img}'}),
                                                 PromptTemplate(input_variables=[
                                                                'bbox_descriptions'], template='{bbox_descriptions}'),
                                                 PromptTemplate(input_variables=['input'], template='{input}')])])

agent = annotate | RunnablePassthrough.assign(
    prediction=format_descriptions | prompt | llm | StrOutputParser() | parse
)

def update_scratchpad(state: AgentState):
    global global_new_tab_url

    old = state.get("scratchpad")
    if old:
        txt = old[0].content
        last_line = txt.rsplit("\n", 1)[-1]
        step = int(re.match(r"\d+", last_line).group()) + 1
    else:
        txt = "Previous action observations:\n"
        step = 1
    txt += f"\n{step}. {state['observation']}"
    if global_new_tab_url:
        state["page"] = global_new_tab_url
    print(state["page"])
    return {**state, "scratchpad": [SystemMessage(content=txt)]}

graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent)
graph_builder.set_entry_point("agent")
graph_builder.add_node("update_scratchpad", update_scratchpad)
graph_builder.add_edge("update_scratchpad", "agent")

tools = {
    "Click": click,
    "Type": type_text,
    "Scroll": scroll,
    "Wait": wait,
    "GoBack": go_back,
    "Google": to_google,
    "Download": download_page,
}

for node_name, tool in tools.items():
    graph_builder.add_node(
        node_name,
        RunnableLambda(tool) | (lambda observation: {
            "observation": observation}),
    )
    graph_builder.add_edge(node_name, "update_scratchpad")

def select_tool(state: AgentState):
    action = state["prediction"]["action"]
    if action == "ANSWER":
        return END
    if action == "retry":
        return "agent"
    return action

graph_builder.add_conditional_edges("agent", select_tool)
graph = graph_builder.compile()

async def call_agent(question: str, page, max_steps: int = 150):
    event_stream = graph.astream(
        {
            "page": page,
            "input": question,
            "scratchpad": [],
        },
        {
            "recursion_limit": max_steps,
        },
    )
    final_answer = None
    steps = []
    img_path = "agent_image.png"
    try:
        async for event in event_stream:
            if "agent" not in event:
                continue
            pred = event["agent"].get("prediction") or {}
            action = pred.get("action")
            action_input = pred.get("args")
            steps.append(f"{len(steps) + 1}. {action}: {action_input}")
            print(steps[-1])

            if "img" in event["agent"]:
                img_data = base64.b64decode(event["agent"]["img"])
                image = PILImage.open(io.BytesIO(img_data))
                image.save(img_path)

            if "ANSWER" in action:
                final_answer = action_input[0] if action_input else "No answer found"
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        try:
            await event_stream.aclose()
        except GeneratorExit:
            print("GeneratorExit exception caught while closing event stream.")
        except Exception as e:
            print(f"An error occurred while closing the event stream: {e}")
    print(f"\n\n Final response: {final_answer}")

async def run_browser():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            accept_downloads=True, 
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://pubmed.ncbi.nlm.nih.gov/",
                "Origin": "https://pubmed.ncbi.nlm.nih.gov",
            }
        )
        page = await context.new_page()
        await page.goto("https://google.com")
        await call_agent("Find you Comparative Performance of Susceptibility Map-Weighted MRI According to the Acquisition Planes in the Diagnosis of Neurodegenerative Parkinsonism paper file on pubmed. Download that PDF file and finish action", page)
        await browser.close()

asyncio.run(run_browser())
