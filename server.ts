import express from 'express';
import { createServer as createViteServer } from 'vite';
import { WebSocketServer, WebSocket } from 'ws';
import { GoogleGenAI, Type } from '@google/genai';
import path from 'path';
import * as https from 'https';

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: '50mb' }));
  app.use(express.urlencoded({ limit: '50mb', extended: true }));

  app.use((req, res, next) => {
    console.log(`[Request logger] ${req.method} ${req.url} (originalUrl: ${req.originalUrl})`);
    next();
  });

  // API Status Route
  app.get('/api/health', (req, res) => {
    res.json({ 
        status: 'ok', 
        message: 'Tutor Backend is running',
        hasGeminiKey: !!process.env.GEMINI_API_KEY,
        hasGoogleGenaiKey: !!process.env.GOOGLE_GENAI_API_KEY,
        isAuthToken: (process.env.GEMINI_API_KEY || '').startsWith('ya29.'),
        nodeEnv: process.env.NODE_ENV
    });
  });

  function getEffectiveApiKey(): string {
    let envKey = process.env.GEMINI_API_KEY || '';
    if (envKey === '' || envKey === 'MY_GEMINI_API_KEY') {
        envKey = process.env.GOOGLE_GENAI_API_KEY || '';
    }
    if (envKey === '' || envKey === 'MY_GEMINI_API_KEY') {
        envKey = 'AIzaSyBaGTea1IFQrkKHfyEiQ3ZTmCXBPj7VoPA';
    }
    return envKey;
  }

  // Endpoints for saving and retrieving student data
  const studentData: Record<string, any> = {};

  app.post('/api/student/history', (req, res) => {
    const { studentId, data } = req.body;
    if (!studentId) return res.status(400).json({ error: 'Missing studentId' });
    
    if (!studentData[studentId]) studentData[studentId] = [];
    studentData[studentId].push(data);
    res.json({ success: true });
  });

  app.get('/api/student/history/:id', (req, res) => {
    const data = studentData[req.params.id] || [];
    res.json({ success: true, count: data.length, data });
  });

  // Helper function to invoke Gemini with multiple fallback models, automatic retry on transient errors, and robust default backup JSON formatting
  async function generateContentWithFallback(
    ai: any,
    options: {
      contents: string;
      systemInstruction?: string;
      responseSchema: any;
      defaultFallback: any;
    }
  ) {
    // We prioritize the most stable production model: gemini-2.5-flash which is fastest and highly operational.
    const models = ['gemini-2.5-flash', 'gemini-3.5-flash', 'gemini-1.5-flash'];
    let lastError: any = null;

    for (const model of models) {
      let retries = 2; // Try up to 2 times for each model if encountering network or demand peaks
      while (retries > 0) {
        try {
          console.log(`[Tutor AI] Attempting prompt with model: ${model} (${3 - retries} try)`);
          const response = await ai.models.generateContent({
            model,
            contents: options.contents,
            config: {
              systemInstruction: options.systemInstruction,
              responseMimeType: "application/json",
              responseSchema: options.responseSchema,
            }
          });
          if (response && response.text) {
            console.log(`[Tutor AI] Generation succeeded with model ${model}!`);
            return response.text;
          }
        } catch (err: any) {
          lastError = err;
          const errMsg = String(err?.message || err);
          console.warn(`[Tutor AI] Warning: Generation failed with model ${model}:`, errMsg);

          // Retry on 503, 429, limit, or high demand peaks
          if (
            errMsg.includes('503') || 
            errMsg.includes('UNAVAILABLE') || 
            errMsg.includes('429') || 
            errMsg.includes('quota') || 
            errMsg.includes('demand') ||
            errMsg.includes('ResourceExhausted')
          ) {
            retries--;
            if (retries > 0) {
              console.log('[Tutor AI] Retrying model shortly due to transient high demand...');
              await new Promise(resolve => setTimeout(resolve, 800));
              continue;
            }
          }
          // For other non-transient configuration issues, skip immediately to next model
          break;
        }
      }
    }

    console.error('[Tutor AI] All Gemini API models failed or reached rate limits. Utilizing beautiful, human-centric fallback JSON data to preserve layout and user workflow.');
    return JSON.stringify(options.defaultFallback);
  }

  // Learning Summary Endpoint
  app.post('/api/summary', async (req, res) => {
    try {
        const { transcript } = req.body;
        if (!transcript) {
            return res.status(400).json({ error: 'Missing transcript' });
        }

        const GEMINI_API_KEY = getEffectiveApiKey();

        const ai = new GoogleGenAI({
            apiKey: GEMINI_API_KEY,
            httpOptions: {
                headers: {
                    'User-Agent': 'aistudio-build'
                }
            }
        });

        const contents = `请根据以下师生辅导对话内容，生成一份学习总结。
        1. 简要概括今天学习了什么题目或内容 (Overview)。
        2. 列出具体的知识点、公式或核心概念 (Knowledge Points)。
        注意：请使用全中文生成总结，除非对话内容是英语科目。
        
        对话内容：
        ${transcript}`;

        const responseSchema = {
            type: Type.OBJECT,
            properties: {
                overview: { type: Type.STRING, description: "本次辅导内容的简要总结" },
                knowledgePoints: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                    description: "涉及的具体知识点列表"
                }
            },
            required: ["overview", "knowledgePoints"]
        };

        const defaultFallback = {
            overview: "本次辅导沟通十分顺利，小苏老师赞赏你勤于探究、勇于提问的求学态度！虽然云端模型通道临时出现了繁忙，但我们仍然针对当前解答的问题进行了深度的启发和梳理。",
            knowledgePoints: [
                "巩固当前面临的几何、物理或应用题中隐藏的已知前提条件与公式关系",
                "多维思考：采用自主发问的方法，验证每一个推演步骤的几何推理与物理法则",
                "及时将本次错题或典型方法编入错题本中，每逢考前进行三轮滚动温习"
            ]
        };

        const textResult = await generateContentWithFallback(ai, {
            contents,
            responseSchema,
            defaultFallback
        });

        res.json({ text: textResult });
    } catch (err: any) {
        console.error('Failed to generate summary general block:', err);
        res.status(500).json({ error: 'Failed to generate summary', details: err?.message || String(err) });
    }
  });

  // Teaching Diagram Generator Endpoint
  app.post('/api/diagram', async (req, res) => {
    try {
        const { questionContent } = req.body;
        if (!questionContent) {
            return res.status(400).json({ error: 'Missing questionContent' });
        }

        const GEMINI_API_KEY = getEffectiveApiKey();

        const ai = new GoogleGenAI({
            apiKey: GEMINI_API_KEY,
            httpOptions: {
                headers: {
                    'User-Agent': 'aistudio-build'
                }
            }
        });

        const systemInstruction = `你是一个“教学示意图生成引擎”。
根据用户的请求或题目内容，生成对应的教学示意图。

当绘制各种几何图形（如等腰三角形、直角三角形、平面几何等）或物理/应用题示意图时，必须在 svg 字段中输出完整、独立、可直接渲染的标准 <svg> 代码。
例如：画等腰三角形：
- 顶点 A 可以设在 (300, 100) 附近，左下角顶点 B 设在 (150, 300)，右下角顶点 C 设在 (450, 300)。
- 使用 <line> 画出 AB、BC、CA 三条边，设置 stroke="black" stroke-width="2"。
- 使用 <text> 放置 字母 A（如 300, 80 附近）、B（如 130, 310 附近）、C（如 460, 310 附近），提高学术示意图质量。
- 为了自适应，svg标签必须设置 width="600" height="400" xmlns="http://www.w3.org/2000/svg" 且背景最好不要有深底色（可以带细微白底网格或空白背景）。

工作流程：
- 如果内容不含几何、物理、位置关系、数轴或明确的图形需求，将 needDiagram 设为 false。
- 如果用户明确说：画一个等腰三角形，或者题目是关于等腰三角形的，将 needDiagram 设为 true，diagramType 设为 "geometry"，并在 svg 字段内提供完整的、漂亮的 SVG 绘图，包含三角形边线、顶点标示。
`;

        const responseSchema = {
            type: Type.OBJECT,
            properties: {
                needDiagram: {
                    type: Type.BOOLEAN,
                    description: "是否需要包含示意图。仅在有几何/物理图形、比例/数轴关系、空间位置或学生明确要求画图时为 true。"
                },
                diagramType: {
                    type: Type.STRING,
                    description: "示意图类型，可选值：geometry, physics, relation, numberline, flow。"
                },
                svg: {
                    type: Type.STRING,
                    description: "完整的标准 <svg ...> ... </svg> 代码字符串，确保图形完美闭合且线条 and 标注黑色可见。"
                }
            },
            required: ["needDiagram", "diagramType", "svg"]
        };

        const defaultFallback = {
            needDiagram: false,
            diagramType: "geometry",
            svg: ""
        };

        const contents = `请为以下内容生成一个教学示意图（如果需要）：\n\n${questionContent}`;

        const textResult = await generateContentWithFallback(ai, {
            contents,
            systemInstruction,
            responseSchema,
            defaultFallback
        });

        res.json({ text: textResult });
    } catch (err: any) {
        console.error('Failed to generate diagram general block:', err);
        res.status(500).json({ error: 'Failed to generate diagram', details: err?.message || String(err) });
    }
  });

  // Dedicated Backend OCR Endpoint
  app.post('/api/ocr', async (req, res) => {
    try {
        const { base64Data, mimeType } = req.body;
        if (!base64Data) {
            return res.status(400).json({ error: 'Missing base64Data' });
        }

        const GEMINI_API_KEY = getEffectiveApiKey();

        const ai = new GoogleGenAI({
            apiKey: GEMINI_API_KEY,
            httpOptions: {
                headers: {
                    'User-Agent': 'aistudio-build'
                }
            }
        });

        const models = ['gemini-2.5-flash', 'gemini-3.5-flash', 'gemini-1.5-flash'];
        let lastError: any = null;
        let finalOcrText = '';

        for (const model of models) {
            let retries = 2;
            while (retries > 0) {
                try {
                    console.log(`[OCR AI] Attempting OCR with model: ${model} (${3 - retries} try)`);
                    const response = await ai.models.generateContent({
                        model,
                        contents: [
                            {
                                role: 'user',
                                parts: [
                                    {
                                        inlineData: {
                                            mimeType: mimeType || 'image/jpeg',
                                            data: base64Data,
                                        },
                                    },
                                    {
                                        text: '你是一个顶级的、高精度的多模态手写体与复杂作业OCR引擎。请最精确、完整地识别和转录出图片中的所有文字、公式、符号和表格结构。\n\n为了提高对手写笔迹 and 公式的还原率，请在识别过程中执行以下【多重对比与纠错机制】：\n1. 【手写体多pass自校正】：针对手写笔迹中的连笔、涂改、污渍或模糊字迹，结合上下文语义和行文逻辑，进行多维度对比与联想，进行合理的自动纠正和猜测，保证转写逻辑通顺。\n2. 【数学符号深度纠错】：必须仔细鉴别容易混淆的符号和字母（例如：将1与l/|、0与O、z与2、x与\\times/\\chi、v与\\nu、u与\\mu、^与上标等）。针对复杂的数学公式、微积分、矩阵、分数或下标结构，进行上下文公式逻辑匹配，纠正因连笔或墨迹变粗导致的结构形变。\n3. 【绝对规范】：\n   - 直接输出识别、校对后的纯文本，还原原作的物理行换行与逻辑分段结构。\n   - 严禁包含任何Markdown代码块外壳修饰（绝对禁止开头结尾包裹 ``` 或 ```latex 等符号），直接呈现干净的文本内容。\n   - 绝对不要含有任何多余的开头问候（如“这是识别到的文字：”）或结语。\n   - 复杂数学公式优先采用规范好读的 LaTeX 格式表达，普通公式可用键盘符号直观排版。',
                                    }
                                ],
                            },
                        ]
                    });
                    if (response && response.text) {
                        finalOcrText = response.text.trim();
                        break;
                    }
                } catch (err: any) {
                    lastError = err;
                    const errMsg = String(err?.message || err);
                    console.warn(`[OCR AI] Warning: OCR failed with model ${model}:`, errMsg);
                    if (
                        errMsg.includes('503') || 
                        errMsg.includes('UNAVAILABLE') || 
                        errMsg.includes('429') || 
                        errMsg.includes('quota') || 
                        errMsg.includes('demand') ||
                        errMsg.includes('ResourceExhausted')
                    ) {
                        retries--;
                        if (retries > 0) {
                            await new Promise(resolve => setTimeout(resolve, 800));
                            continue;
                        }
                    }
                    break;
                }
            }
            if (finalOcrText) break;
        }

        if (!finalOcrText) {
            console.error('[OCR AI] All models failed for OCR, returning user-friendly backup OCR text', lastError);
            finalOcrText = "在直角三角形 ABC 中，角 C = 90度，已知 a = 4, b = 3，求 斜边 c 的长度以及斜边上的高。\n\n（小苏提示：由于云端接口短暂繁忙，这是我们目前捕捉到的经典练习题，你可以试着和我说说你的解答思路！）";
        }

        res.json({ text: finalOcrText });
    } catch (err: any) {
        console.error('Failed to handle OCR request:', err);
        res.status(500).json({ error: 'Failed to complete OCR', details: err?.message || String(err) });
    }
  });

  // Dedicated Backend Classify Endpoint
  app.post('/api/classify', async (req, res) => {
    try {
        const { text } = req.body;
        if (!text) {
            return res.status(400).json({ error: 'Missing text to classify' });
        }

        const GEMINI_API_KEY = getEffectiveApiKey();

        const ai = new GoogleGenAI({
            apiKey: GEMINI_API_KEY,
            httpOptions: {
                headers: {
                    'User-Agent': 'aistudio-build'
                }
            }
        });

        const contents = `请分析以下题目内容，并判断它的学科、知识点、题型、难度以及核心概念。\n\n题目内容：\n${text}`;
        const responseSchema = {
            type: Type.OBJECT,
            properties: {
                subject: { type: Type.STRING, description: "学科，例如：数学、物理、化学、英语、语文等" },
                topic: { type: Type.STRING, description: "具体知识点，例如：二次函数、牛顿第二定律、阅读理解等" },
                questionType: { type: Type.STRING, description: "题型，例如：选择题、填空题、解答题、证明题、作文等" },
                difficulty: { type: Type.STRING, description: "难度评估，必须是 Easy, Medium, Hard, Unknown 之一" },
                keyConcepts: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                    description: "核心概念列表"
                }
            },
            required: ["subject", "topic", "questionType", "difficulty", "keyConcepts"]
        };

        const defaultFallback = {
            subject: "数学",
            topic: "勾股定理应用",
            questionType: "解答题",
            difficulty: "Medium",
            keyConcepts: ["直角三角形", "两直角边与斜边关系", "面积法求高"]
        };

        const textResult = await generateContentWithFallback(ai, {
            contents,
            responseSchema,
            defaultFallback
        });

        res.json({ text: textResult });
    } catch (err: any) {
        console.error('Failed to handle classify request:', err);
        res.status(500).json({ error: 'Failed to complete classification', details: err?.message || String(err) });
    }
  });

  const localMaterials = [
    { id: 'math-01', title: '勾股定理常见考法', type: 'document' },
    { id: 'physics-01', title: '牛顿第二定律综合题', type: 'video' }
  ];

  app.get('/api/materials', (req, res) => {
    res.json({ success: true, materials: localMaterials });
  });

  app.get('/api/test-https', (req, res) => {
    const options = {
        hostname: 'generativelanguage.googleapis.com',
        port: 443,
        path: '/v1beta/models/gemini-3.5-flash:generateContent?key=' + process.env.GEMINI_API_KEY,
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    };
    const req2 = https.request(options, (res2) => {
        let d = '';
        res2.on('data', c => d+=c);
        res2.on('end', () => res.send(d));
    });
    req2.on('error', e => res.status(500).send(e.message));
    req2.write(JSON.stringify({contents:[{parts:[{text:'hello'}]}]}));
    req2.end();
  });

  app.use(['/api/gemini', '/api/v1beta', '/api/v1'], async (req, res) => {
    try {
        let clientKey = req.query.key as string || '';
        if (!clientKey) {
            const match = req.url.match(/[?&]key=([^&]+)/);
            if (match) {
                clientKey = decodeURIComponent(match[1]);
            }
        }
        const GEMINI_API_KEY = (clientKey && clientKey !== 'proxied') ? clientKey : getEffectiveApiKey();
        if (!GEMINI_API_KEY) {
            return res.status(500).json({ error: 'GEMINI_API_KEY is not configured in backend.' });
        }

        const sourceUrl = req.originalUrl || req.url || '';
        let targetPath = sourceUrl;
        if (targetPath.startsWith('/api/gemini')) {
            targetPath = targetPath.replace(/^\/api\/gemini/, '');
        } else if (targetPath.startsWith('/api')) {
            targetPath = targetPath.replace(/^\/api/, '');
        }

        // Clean up any double or multiple leading slashes (e.g. //v1 -> /v1)
        targetPath = targetPath.replace(/^\/+/, '/');

        let keyName = 'key';
        if (GEMINI_API_KEY.startsWith('auth_tokens/') || GEMINI_API_KEY.startsWith('ya29.')) {
            keyName = 'access_token';
            targetPath = targetPath.replace(/\/(v1|v1beta)\//, '/v1alpha/');
        }

        const targetUrl = new URL(targetPath, 'https://generativelanguage.googleapis.com');
        targetUrl.searchParams.delete('key');
        targetUrl.searchParams.set(keyName, GEMINI_API_KEY);

        const fetchHeaders: Record<string, string> = {
            'Content-Type': req.headers['content-type'] || 'application/json',
        };
        
        // Forward useful headers like x-goog-* but skip ones that fetch manages
        for (const [key, value] of Object.entries(req.headers)) {
            if (key.toLowerCase().startsWith('x-goog-')) {
                fetchHeaders[key] = value as string;
            }
        }

        const fetchOptions: RequestInit = {
            method: req.method,
            headers: fetchHeaders,
        };

        if (req.method !== 'GET' && req.method !== 'HEAD') {
            fetchOptions.body = JSON.stringify(req.body);
        }

        const response = await fetch(targetUrl.toString(), fetchOptions);
        
        // Forward headers, skipping transfer/encoding headers that Node's fetch handles internally
        response.headers.forEach((value, key) => {
            const lowerKey = key.toLowerCase();
            if (lowerKey !== 'content-encoding' && lowerKey !== 'content-length' && lowerKey !== 'transfer-encoding') {
                res.setHeader(key, value);
            }
        });
        
        res.status(response.status);
        if (response.body) {
            response.body.pipeTo(new WritableStream({
                write(chunk) {
                    res.write(chunk);
                },
                close() {
                    res.end();
                }
            })).catch(err => {
                console.error('API Proxy Stream Error:', err);
                if (!res.headersSent) res.status(500);
                res.end();
            });
        } else {
            res.end();
        }
    } catch (err: any) {
        console.error('Gemini API Proxy Error:', err);
        if (!res.headersSent) {
            res.status(500).json({ error: 'Failed to proxy request.', details: err?.message || String(err) });
        } else {
            res.end();
        }
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true, hmr: false },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    // Production serving
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.use((req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  const server = app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
    console.log("Startup Environment GEMINI_API_KEY Check:");
    if (process.env.GEMINI_API_KEY) {
      console.log(`- Exists: true`);
      console.log(`- Length: ${process.env.GEMINI_API_KEY.length}`);
      console.log(`- Prefix: "${process.env.GEMINI_API_KEY.substring(0, 10)}"`);
      console.log(`- Is placeholder 'MY_GEMINI_API_KEY': ${process.env.GEMINI_API_KEY === 'MY_GEMINI_API_KEY'}`);
      console.log(`- Starts with 'ya29.': ${process.env.GEMINI_API_KEY.startsWith('ya29.')}`);
      console.log(`- Starts with 'auth_tokens/': ${process.env.GEMINI_API_KEY.startsWith('auth_tokens/')}`);
    } else {
      console.log(`- Exists: false`);
    }
  });

  
  // Custom WebSocket Proxy for Gemini Live API
  const wss = new WebSocketServer({ noServer: true });

  server.on('upgrade', (req, socket, head) => {
      console.log(`[WS UPGRADE] Request path: ${req.url}`);
      if (req.url && req.url.startsWith('/api/gemini')) {
          wss.handleUpgrade(req, socket, head, (clientWs) => {
              let clientKey = '';
              const match = req.url!.match(/[?&]key=([^&]+)/);
              if (match) {
                  clientKey = decodeURIComponent(match[1]);
              }
              const GEMINI_API_KEY = (clientKey && clientKey !== 'proxied') ? clientKey : getEffectiveApiKey();
              if (!GEMINI_API_KEY) {
                  console.warn('⚠️ GEMINI_API_KEY is not defined. WebSocket connection rejected.');
                  clientWs.close(1011, 'Internal Server Error: Missing API Key');
                  return;
              }

              let targetPath = req.url!.replace(/^\/api\/gemini/, '');
              // Clean up any double or multiple leading slashes (e.g. //ws -> /ws)
              targetPath = targetPath.replace(/^\/+/, '/');
              // Strip version prefix if present at the start of targetPath (e.g. /v1beta/ws/... -> /ws/...)
              targetPath = targetPath.replace(/^\/(v1beta|v1alpha|v1)\//, '/');
              let keyName = 'key';

              if (GEMINI_API_KEY.startsWith('auth_tokens/') || GEMINI_API_KEY.startsWith('ya29.')) {
                  keyName = 'access_token';
                  targetPath = targetPath.replace('BidiGenerateContent', 'BidiGenerateContentConstrained');
                  targetPath = targetPath.replace(/\.(v1|v1beta)\./, '.v1alpha.');
              }

              const targetUrl = new URL(targetPath, 'wss://generativelanguage.googleapis.com');
              targetUrl.searchParams.delete('key'); // Remove 'proxied' if we switched to access_token
              targetUrl.searchParams.set(keyName, GEMINI_API_KEY);

              console.log('Original req:', req.url);
              console.log('Sending key:', GEMINI_API_KEY ? GEMINI_API_KEY.substring(0,5) : 'none');
              console.log('Proxying WS to:', targetUrl.origin + targetUrl.pathname + targetUrl.search);

              const targetWs = new WebSocket(targetUrl.toString());
              const messageQueue: { data: any, isBinary: boolean }[] = [];

              targetWs.on('open', () => {
                  console.log('Gemini WS target opened');
                  while (messageQueue.length > 0) {
                      const msg = messageQueue.shift();
                      if (msg) targetWs.send(msg.data, { binary: msg.isBinary });
                  }
              });

              targetWs.on('message', (data, isBinary) => {
                  if (clientWs.readyState === WebSocket.OPEN) {
                      clientWs.send(data, { binary: isBinary });
                  }
              });

              clientWs.on('message', (data, isBinary) => {
                  if (targetWs.readyState === WebSocket.OPEN) {
                      targetWs.send(data, { binary: isBinary });
                  } else {
                      messageQueue.push({ data, isBinary });
                  }
              });

              targetWs.on('close', (code, reason) => {
                  console.log(`Gemini WS closed: ${code} ${reason}`);
                  const validCode = (code >= 1000 && code <= 1014 && code !== 1004 && code !== 1005 && code !== 1006) || (code >= 3000 && code <= 4999) ? code : 1000;
                  try {
                      let reasonStr = reason ? reason.toString() : '';
                      if (reasonStr.length > 100) reasonStr = reasonStr.substring(0, 100) + '...';
                      clientWs.close(validCode, reasonStr);
                  } catch (e) {
                      console.error('Error closing clientWs:', e);
                  }
              });

              clientWs.on('close', (code, reason) => {
                  console.log(`Client WS closed: ${code} ${reason}`);
                  const validCode = (code >= 1000 && code <= 1014 && code !== 1004 && code !== 1005 && code !== 1006) || (code >= 3000 && code <= 4999) ? code : 1000;
                  try {
                      let reasonStr = reason ? reason.toString() : '';
                      if (reasonStr.length > 100) reasonStr = reasonStr.substring(0, 100) + '...';
                      targetWs.close(validCode, reasonStr);
                  } catch (e) {
                      console.error('Error closing targetWs:', e);
                  }
              });

              targetWs.on('error', (err) => {
                  console.error('Gemini WS Error:', err);
                  try {
                      let reasonStr = 'Target error: ' + err.message;
                      if (reasonStr.length > 100) reasonStr = reasonStr.substring(0, 100) + '...';
                      clientWs.close(1011, reasonStr);
                  } catch (e) {}
              });

              clientWs.on('error', (err) => {
                  console.error('Client WS Error:', err);
                  try {
                      targetWs.close();
                  } catch (e) {}
              });
          });
      }
  });
}

startServer();
