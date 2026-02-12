from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs

from pos_models import tokenize, train_bundle

MODELS = {
    "en": train_bundle("en"),
    "zh": train_bundle("zh"),
}


def render_page(text: str = "", lang: str = "zh", method: str = "structured_perceptron", result=None) -> str:
    result = result or []
    metric_rows = {
        language: {
            "structured_perceptron": f"{bundle.metrics['structured_perceptron'] * 100:.2f}%",
            "hmm_baseline": f"{bundle.metrics['hmm_baseline'] * 100:.2f}%",
        }
        for language, bundle in MODELS.items()
    }

    tags_html = (
        "".join([f"<span class='pill'>{token} / <b>{tag}</b></span>" for token, tag in result])
        if result
        else "<p class='muted'>提交句子后将在此显示词性标注结果。</p>"
    )

    return f"""<!doctype html>
<html lang='zh-CN'>
<head>
<meta charset='UTF-8'>
<meta name='viewport' content='width=device-width, initial-scale=1.0'>
<title>结构化感知机词性标注演示</title>
<style>
body {{font-family: Arial, sans-serif; background: #f6f8fb; margin: 0;}}
.container {{max-width: 960px; margin: 30px auto; padding: 0 16px;}}
.card {{background: #fff; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,.08); margin-bottom: 16px;}}
textarea {{width: 100%; min-height: 100px; padding: 10px; border-radius: 8px; border: 1px solid #ccc;}}
select, button {{padding: 8px 12px; margin-right: 8px; border-radius: 8px; border: 1px solid #bbb;}}
button {{background: #2f6feb; color: white; border: none; cursor: pointer;}}
table {{width: 100%; border-collapse: collapse; margin-top: 10px;}}
th, td {{border: 1px solid #ddd; padding: 8px; text-align: center;}}
th {{background: #eef3ff;}}
.pill {{display:inline-block; background:#eef3ff; margin:3px; padding:6px 9px; border-radius:999px;}}
.muted {{color:#666; font-size:14px;}}
</style>
</head>
<body>
<div class='container'>
<div class='card'>
<h2>中英文词性标注（结构化感知机）</h2>
<p class='muted'>输入原句（英文可直接输入，中文建议空格分词；若不分词，系统按字切分）。</p>
<form method='post'>
<textarea name='text' placeholder='例如：我 爱 自然 语言 处理'>{text}</textarea>
<div style='margin-top:10px;'>
<label>语言：</label>
<select name='lang'>
  <option value='zh' {'selected' if lang == 'zh' else ''}>中文</option>
  <option value='en' {'selected' if lang == 'en' else ''}>English</option>
</select>
<label>方法：</label>
<select name='method'>
  <option value='structured_perceptron' {'selected' if method == 'structured_perceptron' else ''}>结构化感知机</option>
  <option value='hmm_baseline' {'selected' if method == 'hmm_baseline' else ''}>HMM 基线</option>
</select>
<button type='submit'>开始标注</button>
</div>
</form>
</div>
<div class='card'><h3>标注结果</h3>{tags_html}</div>
<div class='card'>
<h3>方法比较（内置测试集准确率）</h3>
<table>
<tr><th>语言</th><th>结构化感知机</th><th>HMM 基线</th></tr>
<tr><td>中文</td><td>{metric_rows['zh']['structured_perceptron']}</td><td>{metric_rows['zh']['hmm_baseline']}</td></tr>
<tr><td>English</td><td>{metric_rows['en']['structured_perceptron']}</td><td>{metric_rows['en']['hmm_baseline']}</td></tr>
</table>
<p class='muted'>优点：结构化感知机直接在序列层面优化，特征灵活；缺点：依赖人工特征，数据量小或未登录词多时泛化受限。</p>
<p class='muted'>HMM 优点：实现简单、可解释；缺点：独立性假设较强，通常精度低于判别式模型。</p>
</div>
</div>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def _send_html(self, html: str):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send_html(render_page())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length).decode("utf-8")
        form = parse_qs(payload)

        text = form.get("text", [""])[0]
        lang = form.get("lang", ["zh"])[0]
        method = form.get("method", ["structured_perceptron"])[0]

        tokens = tokenize(text, lang)
        bundle = MODELS.get(lang, MODELS["zh"])
        model = bundle.perceptron if method == "structured_perceptron" else bundle.hmm
        tags = model.decode(tokens)

        self._send_html(render_page(text=text, lang=lang, method=method, result=list(zip(tokens, tags))))


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8000), Handler)
    print("Server running: http://127.0.0.1:8000")
    server.serve_forever()
