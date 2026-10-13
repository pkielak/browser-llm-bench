// llm-bench dashboard server.
// Serves dashboard/ (static), results/ (static), plus:
//   GET  /api/results          → JSON list of result files
//   POST /api/results          → {filename, token?, content} writes results/<filename>
//                                (token: if set, also pushes the repo via Gitea API…
//                                 no — keeps it simple: local write only, user pushes)
// Mount: /srv → llm-bench repo root (dashboard/, results/ must exist under it).
import http from "node:http";
import fs from "node:fs";
import path from "node:path";

const ROOT = process.env.SRV_ROOT || "/srv";
const DASHBOARD = path.join(ROOT, "dashboard");
const RESULTS = path.join(ROOT, "results");
const PORT = Number(process.env.PORT || 8080);

const MIME = {
  ".html": "text/html", ".json": "application/json", ".js": "text/javascript",
  ".css": "text/css", ".svg": "image/svg+xml", ".png": "image/png",
};

function send(res, code, body, type = "application/json") {
  res.writeHead(code, { "content-type": type });
  res.end(body);
}

function serveFile(res, file) {
  fs.readFile(file, (err, data) => {
    if (err) return send(res, 404, JSON.stringify({ error: "not found" }));
    send(res, 200, data, MIME[path.extname(file)] || "application/octet-stream");
  });
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, "http://localhost");
  const p = url.pathname;

  if (p === "/api/results" && req.method === "GET") {
    const files = fs.existsSync(RESULTS)
      ? fs.readdirSync(RESULTS).filter((f) => f.endsWith(".json")).sort()
      : [];
    return send(res, 200, JSON.stringify(files));
  }

  if (p === "/api/results" && req.method === "POST") {
    let body = "";
    req.on("data", (c) => { body += c; if (body.length > 5e6) req.destroy(); });
    req.on("end", () => {
      try {
        const { filename, content } = JSON.parse(body);
        // sanitize: basename only, [a-zA-Z0-9._-]+
        let name = path.basename(String(filename || "")).replace(/[^a-zA-Z0-9._-]/g, "_");
        if (!name) throw new Error("filename required");
        name = name.replace(/\.json$/, "") + ".json";
        const text = JSON.stringify(content, null, 2);
        JSON.parse(text); // validate
        fs.mkdirSync(RESULTS, { recursive: true });
        fs.writeFileSync(path.join(RESULTS, name), text + "\n");
        return send(res, 200, JSON.stringify({ path: name }));
      } catch (e) {
        return send(res, 400, JSON.stringify({ error: e.message }));
      }
    });
    return;
  }

  if (p.startsWith("/results/")) {
    const file = path.join(RESULTS, path.basename(p));
    if (!file.startsWith(RESULTS)) return send(res, 400, JSON.stringify({ error: "bad path" }));
    return serveFile(res, file);
  }

  // everything else → dashboard static
  let file = path.join(DASHBOARD, p === "/" ? "index.html" : p);
  if (!file.startsWith(DASHBOARD)) return send(res, 400, JSON.stringify({ error: "bad path" }));
  if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) file = path.join(DASHBOARD, "index.html");
  serveFile(res, file);
});

server.listen(PORT, () => console.log(`llm-bench dashboard on :${PORT} (root ${ROOT})`));
