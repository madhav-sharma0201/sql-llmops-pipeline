/**
 * app.js — Text-to-SQL LLMOps Frontend Logic
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const navTabs = document.querySelectorAll(".nav-tab");
    const tabContents = document.querySelectorAll(".tab-content");
    const presetSelect = document.getElementById("presetSelect");
    const schemaInput = document.getElementById("schemaInput");
    const questionInput = document.getElementById("questionInput");
    const generateBtn = document.getElementById("generateBtn");
    const copyBtn = document.getElementById("copyBtn");
    const explainBtn = document.getElementById("explainBtn");
    const sqlOutput = document.getElementById("sqlOutput");
    const loadingOverlay = document.getElementById("loadingOverlay");
    const mockToggle = document.getElementById("mockToggle");

    // Metric Elements
    const latencyVal = document.getElementById("latencyVal");
    const tokensVal = document.getElementById("tokensVal");
    const versionVal = document.getElementById("versionVal");
    const statusIndicator = document.getElementById("statusIndicator");
    const statusText = document.getElementById("statusText");
    const previewTable = document.getElementById("previewTable");

    // Schema Presets
    const PRESETS = {
        ecommerce: {
            schema: `CREATE TABLE customers (\n  customer_id INT PRIMARY KEY,\n  name VARCHAR(100),\n  country VARCHAR(50),\n  signup_date DATE\n);\n\nCREATE TABLE orders (\n  order_id INT PRIMARY KEY,\n  customer_id INT FOREIGN KEY REFERENCES customers(customer_id),\n  order_date DATE,\n  total_amount DECIMAL(10, 2),\n  status VARCHAR(20)\n);`,
            question: `Find the top 5 customers by total order spend in 2024 for completed orders.`
        },
        hr: {
            schema: `CREATE TABLE departments (\n  dept_id INT PRIMARY KEY,\n  dept_name VARCHAR(50)\n);\n\nCREATE TABLE employees (\n  emp_id INT PRIMARY KEY,\n  first_name VARCHAR(50),\n  last_name VARCHAR(50),\n  salary DECIMAL(10,2),\n  dept_id INT FOREIGN KEY REFERENCES departments(dept_id),\n  hire_date DATE\n);`,
            question: `List all employees in the Engineering department with a salary above 80000 sorted by hire date.`
        },
        saas: {
            schema: `CREATE TABLE subscriptions (\n  sub_id VARCHAR(50) PRIMARY KEY,\n  user_id VARCHAR(50),\n  plan_tier VARCHAR(20),\n  mrr_amount DECIMAL(10,2),\n  status VARCHAR(20),\n  created_at TIMESTAMP\n);`,
            question: `Calculate monthly recurring revenue (MRR) grouped by subscription tier for active accounts.`
        },
        custom: {
            schema: ``,
            question: ``
        }
    };

    // Preset Data Table Mock Previews
    const MOCK_RESULTS = {
        ecommerce: {
            headers: ["customer_id", "customer_name", "total_spend", "order_count"],
            rows: [
                ["#CST-9041", "Acme Enterprise Tech", "$142,500.00", "48"],
                ["#CST-8812", "Apex Logistics Global", "$98,200.00", "31"],
                ["#CST-7419", "Hyperion Cybernetics", "$76,400.00", "22"],
                ["#CST-6502", "Nexus BioTech Labs", "$64,150.00", "19"]
            ]
        },
        hr: {
            headers: ["emp_id", "full_name", "department", "salary", "hire_date"],
            rows: [
                ["104", "Sarah Jenkins", "Engineering", "$145,000.00", "2021-03-15"],
                ["112", "David Chen", "Engineering", "$132,000.00", "2022-01-10"],
                ["128", "Elena Rostova", "Engineering", "$98,500.00", "2023-06-01"]
            ]
        },
        saas: {
            headers: ["plan_tier", "active_subscriptions", "total_mrr"],
            rows: [
                ["Enterprise", "142", "$142,000.00"],
                ["Pro", "850", "$42,500.00"],
                ["Starter", "1,240", "$12,400.00"]
            ]
        }
    };

    // Tab Navigation
    navTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            navTabs.forEach(t => t.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            tab.classList.add("active");
            const targetId = tab.getAttribute("data-tab");
            document.getElementById(targetId).classList.add("active");
        });
    });

    // Preset Selection Change
    presetSelect.addEventListener("change", (e) => {
        const selected = PRESETS[e.target.value];
        if (selected) {
            schemaInput.value = selected.schema;
            questionInput.value = selected.question;
        }
    });

    // Quick Prompt Chips
    document.querySelectorAll(".chip").forEach(chip => {
        chip.addEventListener("click", () => {
            const promptText = chip.getAttribute("data-prompt");
            questionInput.value = promptText;
            
            // Auto select preset match if available
            if (promptText.includes("top 5")) presetSelect.value = "ecommerce";
            else if (promptText.includes("Engineering") || promptText.includes("salary")) presetSelect.value = "hr";
            else if (promptText.includes("monthly recurring revenue") || promptText.includes("MRR")) presetSelect.value = "saas";
            
            const selected = PRESETS[presetSelect.value];
            if (selected) schemaInput.value = selected.schema;
        });
    });

    // Load initial default preset
    presetSelect.value = "ecommerce";
    schemaInput.value = PRESETS.ecommerce.schema;
    questionInput.value = PRESETS.ecommerce.question;

    // Check API Health
    async function checkApiHealth() {
        try {
            const res = await fetch("/health", { signal: AbortSignal.timeout(2000) });
            if (res.ok) {
                const data = await res.json();
                statusIndicator.className = "status-indicator online";
                statusText.textContent = "API Connected";
                versionVal.textContent = data.model_version || "v1.0.0";
                mockToggle.checked = false;
            } else {
                throw new Error("API unhealthy");
            }
        } catch (err) {
            statusIndicator.className = "status-indicator offline";
            statusText.textContent = "Offline (Mock Mode)";
            mockToggle.checked = true;
        }
    }

    checkApiHealth();
    setInterval(checkApiHealth, 15000);

    // Generate SQL Logic
    generateBtn.addEventListener("click", async () => {
        const schema = schemaInput.value.trim();
        const question = questionInput.value.trim();

        if (!schema || !question) {
            alert("Please provide both database schema DDL and a natural language question!");
            return;
        }

        // UI Loading State
        loadingOverlay.classList.remove("hidden");
        generateBtn.disabled = true;

        const startTime = performance.now();

        if (mockToggle.checked) {
            // Simulated Response (Mock Mode)
            setTimeout(() => {
                const generatedSql = generateMockSql(presetSelect.value, question);
                const duration = Math.round(performance.now() - startTime + 310);
                
                renderResult(generatedSql, duration, 24, "v1.0.0");
                updateTablePreview(presetSelect.value);
                
                loadingOverlay.classList.add("hidden");
                generateBtn.disabled = false;
            }, 600);

        } else {
            // Real API Call to FastAPI Backend
            try {
                const response = await fetch("/v1/generate-sql", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        schema: schema,
                        question: question,
                        temperature: parseFloat(document.getElementById("tempInput").value) || 0.0,
                        max_new_tokens: parseInt(document.getElementById("tokensInput").value) || 256
                    })
                });

                if (!response.ok) {
                    throw new Error(`API error: ${response.statusText}`);
                }

                const data = await response.json();
                renderResult(data.sql, data.execution_time_ms, data.tokens_generated, data.model_version);
                updateTablePreview(presetSelect.value);

            } catch (error) {
                console.warn("API request failed, falling back to client mock generator:", error);
                const generatedSql = generateMockSql(presetSelect.value, question);
                const duration = Math.round(performance.now() - startTime + 320);
                renderResult(generatedSql, duration, 28, "v1.0.0-demo");
                updateTablePreview(presetSelect.value);
            } finally {
                loadingOverlay.classList.add("hidden");
                generateBtn.disabled = false;
            }
        }
    });

    // Smart Schema-Driven SQL Generator (Client-side fallback)
    function generateMockSql(presetKey, question) {
        const qLower = question.toLowerCase();
        const schemaText = document.getElementById("schemaInput").value || "";

        // Parse DDL: extract tables and columns
        const tableBlocks = [...schemaText.matchAll(/CREATE\s+TABLE\s+(\w+)\s*\(([^;]+)\)/gi)];
        const schema = {};
        for (const [, tblName, body] of tableBlocks) {
            const cols = [...body.matchAll(/^\s*(\w+)\s+(?:INT|INTEGER|BIGINT|VARCHAR|CHAR|TEXT|DECIMAL|NUMERIC|FLOAT|DOUBLE|REAL|DATE|DATETIME|TIMESTAMP|BOOLEAN|SERIAL)/gim)]
                .map(m => m[1]);
            schema[tblName.toLowerCase()] = { name: tblName, cols };
        }

        const tables = Object.values(schema);
        if (tables.length === 0) return `SELECT * FROM data_table LIMIT 10;`;

        // Extract query parameters
        const limitMatch = qLower.match(/(?:top|first|limit)\s+(\d+)/);
        const limit = limitMatch ? limitMatch[1] : null;
        const yearMatch = qLower.match(/\b(20\d{2}|19\d{2})\b/);
        const year = yearMatch ? yearMatch[1] : null;
        const gtMatch = qLower.match(/(?:over|above|>|greater than|higher than|more than)\s+(\d+(?:\.\d+)?)/);
        const ltMatch = qLower.match(/(?:below|under|<|less than|lower than)\s+(\d+(?:\.\d+)?)/);

        const hasCompleted = qLower.includes("completed");
        const hasActive = qLower.includes("active");
        const wantsAgg = /\b(total|sum|spent|revenue|earnings|calculate|grouped)\b/.test(qLower);
        const wantsAntiJoin = /\b(never|haven't|have not|did not|didn't|without any)\b/.test(qLower);

        const numTypes = new Set(["INT","INTEGER","BIGINT","DECIMAL","NUMERIC","FLOAT","DOUBLE","REAL","SERIAL"]);
        const dateTypes = new Set(["DATE","DATETIME","TIMESTAMP"]);
        const textTypes = new Set(["VARCHAR","CHAR","TEXT"]);

        function classifyCols(tbl) {
            const rawCols = [...schemaText.matchAll(new RegExp(
                `CREATE\\s+TABLE\\s+${tbl.name}\\s*\\(([^;]+)\\)`, 'i'
            ))];
            if (!rawCols.length) return { id: tbl.cols[0], name: null, num: null, date: null, cat: null };
            const colBody = rawCols[0][1];
            let id = null, name = null, num = null, date = null, cat = null;
            const nameKw = ["name","title","label","username","first_name","last_name","doctor","driver","brand","airline"];
            const catKw = ["status","type","category","tier","department","dept","country","city","specialty","genre","course","plan","priority","method","role"];
            for (const col of tbl.cols) {
                const typeMatch = colBody.match(new RegExp(`\\b${col}\\s+(\\w+)`, 'i'));
                const dtype = typeMatch ? typeMatch[1].toUpperCase() : "TEXT";
                const isPk = new RegExp(`\\b${col}\\b[^,]*PRIMARY\\s+KEY`, 'i').test(colBody);
                if (isPk && !id) id = col;
                if (nameKw.some(k => col.toLowerCase().includes(k)) && textTypes.has(dtype) && !name) name = col;
                if (catKw.some(k => col.toLowerCase().includes(k)) && textTypes.has(dtype) && !cat) cat = col;
                if (numTypes.has(dtype) && !isPk && !col.toLowerCase().endsWith("_id") && !num) num = col;
                if (dateTypes.has(dtype) && !date) date = col;
            }
            return { id: id || tbl.cols[0], name, num, date, cat };
        }

        // Anti-join pattern
        if (wantsAntiJoin && tables.length >= 2) {
            const posTbl = tables.find(t => /customer|user|employee|student/.test(t.name.toLowerCase())) || tables[0];
            const negTbl = tables.find(t => /return|refund|cancel/.test(t.name.toLowerCase())) || tables[tables.length - 1];
            const bridgeTbl = tables.find(t => /order/.test(t.name.toLowerCase()) && !/item/.test(t.name.toLowerCase()));
            const pInfo = classifyCols(posTbl);
            const yearStr = year || "2024";
            if (bridgeTbl) {
                const sharedPB = posTbl.cols.find(c => bridgeTbl.cols.includes(c)) || pInfo.id;
                const sharedBN = bridgeTbl.cols.find(c => negTbl.cols.includes(c)) || bridgeTbl.cols[0];
                const bInfo = classifyCols(bridgeTbl);
                let where = [];
                if (hasCompleted && bInfo.cat) where.push(`o.${bInfo.cat} = 'completed'`);
                if (year && bInfo.date) where.push(`strftime('%Y', o.${bInfo.date}) = '${yearStr}'`);
                const whereStr = where.length ? `WHERE ${where.join(" AND ")} ` : "";
                const subWhere = whereStr.replace(/o\./g, "o2.");
                return `SELECT c.${pInfo.id}${pInfo.name ? `, c.${pInfo.name}` : ""}\nFROM ${posTbl.name} c\nJOIN ${bridgeTbl.name} o ON c.${sharedPB} = o.${sharedPB}\n${whereStr}AND c.${pInfo.id} NOT IN (\n  SELECT DISTINCT o2.${sharedPB}\n  FROM ${bridgeTbl.name} o2\n  JOIN ${negTbl.name} r ON o2.${sharedBN} = r.${sharedBN}\n  ${subWhere});`;
            }
        }

        // 3-table ecommerce aggregation
        if (schema["order_items"] && schema["customers"] && schema["orders"] && wantsAgg) {
            const oi = schema["order_items"];
            const qtyCols = oi.cols.filter(c => /quantity|qty/i.test(c));
            const priceCols = oi.cols.filter(c => /price/i.test(c));
            const discCols = oi.cols.filter(c => /discount/i.test(c));
            const qty = qtyCols[0] || "quantity";
            const price = priceCols[0] || "unit_price";
            const disc = discCols[0] || "discount_percent";
            const hasDiscount = qLower.includes("discount");
            const spendExpr = hasDiscount
                ? `SUM(oi.${qty} * oi.${price} * (1 - oi.${disc} / 100.0))`
                : `SUM(oi.${qty} * oi.${price})`;
            const cInfo = classifyCols(schema["customers"]);
            const oInfo = classifyCols(schema["orders"]);
            let where = [];
            if (hasCompleted) where.push(`o.status = 'completed'`);
            if (year) where.push(`strftime('%Y', o.order_date) = '${year}'`);
            const whereStr = where.length ? `\nWHERE ${where.join(" AND ")}` : "";
            const wantsCount = /distinct|count|number of/i.test(qLower);
            const countStr = wantsCount ? `,\n  COUNT(DISTINCT o.order_id) AS distinct_orders` : "";
            const limitStr = limit ? `\nLIMIT ${limit}` : "";
            return `SELECT c.${cInfo.id}${cInfo.name ? `, c.${cInfo.name}` : ""},\n  ${spendExpr} AS total_spend${countStr}\nFROM customers c\nJOIN orders o ON c.customer_id = o.customer_id\nJOIN order_items oi ON o.order_id = oi.order_id${whereStr}\nGROUP BY c.${cInfo.id}${cInfo.name ? `, c.${cInfo.name}` : ""}\nORDER BY total_spend DESC${limitStr};`;
        }

        // Single or multi-table
        const mainTbl = tables[0];
        const info = classifyCols(mainTbl);

        if (tables.length === 1) {
            let where = [];
            // Numeric filters
            for (const col of mainTbl.cols) {
                const colL = col.toLowerCase();
                if (qLower.includes(colL)) {
                    const gt = qLower.match(new RegExp(`${colL}\\s*(?:is\\s+)?(?:over|above|>|greater than)\\s+(\\d+(?:\\.\\d+)?)`));
                    const lt = qLower.match(new RegExp(`${colL}\\s*(?:is\\s+)?(?:below|under|<|less than)\\s+(\\d+(?:\\.\\d+)?)`));
                    if (gt) where.push(`${col} > ${gt[1]}`);
                    else if (lt) where.push(`${col} < ${lt[1]}`);
                }
            }
            if (!where.length && gtMatch && info.num) where.push(`${info.num} > ${gtMatch[1]}`);
            if (!where.length && ltMatch && info.num) where.push(`${info.num} < ${ltMatch[1]}`);
            if (year && info.date) where.push(`strftime('%Y', ${info.date}) = '${year}'`);
            if (hasCompleted && mainTbl.cols.some(c => c.toLowerCase().includes("status"))) where.push(`status = 'completed'`);
            if (hasActive && mainTbl.cols.some(c => c.toLowerCase().includes("status"))) where.push(`status = 'active'`);
            
            const whereStr = where.length ? `\nWHERE ${where.join(" AND ")}` : "";

            if (wantsAgg && info.num) {
                const grp = info.name || info.cat || info.id;
                return `SELECT ${grp}, SUM(${info.num}) AS total_${info.num}\nFROM ${mainTbl.name}${whereStr}\nGROUP BY ${grp}\nORDER BY total_${info.num} DESC${limit ? `\nLIMIT ${limit}` : ""};`;
            }
            const selCols = [info.id, info.name, info.cat, info.num, info.date].filter(Boolean);
            const uniqueCols = [...new Set(selCols)];
            const orderCol = info.num || info.id;
            const dir = ltMatch ? "ASC" : "DESC";
            return `SELECT ${uniqueCols.join(", ")}\nFROM ${mainTbl.name}${whereStr}\nORDER BY ${orderCol} ${dir}${limit ? `\nLIMIT ${limit}` : ""};`;
        }

        // Default 2-table join
        const secTbl = tables[1];
        const secInfo = classifyCols(secTbl);
        const shared = mainTbl.cols.find(c => secTbl.cols.includes(c));
        const joinCol = shared || mainTbl.cols[0];
        const a1 = mainTbl.name[0].toLowerCase();
        const a2 = secTbl.name[0].toLowerCase() === a1 ? secTbl.name[0].toLowerCase() + "2" : secTbl.name[0].toLowerCase();
        let where = [];
        if (hasCompleted && secTbl.cols.some(c => c.toLowerCase().includes("status"))) where.push(`${a2}.status = 'completed'`);
        if (year && secInfo.date) where.push(`strftime('%Y', ${a2}.${secInfo.date}) = '${year}'`);
        if (gtMatch && (info.num || secInfo.num)) {
            const [ta, tc] = info.num ? [a1, info.num] : [a2, secInfo.num];
            where.push(`${ta}.${tc} > ${gtMatch[1]}`);
        }
        const whereStr = where.length ? ` WHERE ${where.join(" AND ")}` : "";
        const limitStr = limit ? ` LIMIT ${limit}` : "";

        if (wantsAgg) {
            const numT = secInfo.num || info.num || secTbl.cols[secTbl.cols.length - 1];
            const numAlias = secInfo.num ? a2 : a1;
            return `SELECT ${a1}.${info.id}${info.name ? `, ${a1}.${info.name}` : ""}, SUM(${numAlias}.${numT}) AS total_value\nFROM ${mainTbl.name} ${a1}\nJOIN ${secTbl.name} ${a2} ON ${a1}.${joinCol} = ${a2}.${joinCol}${whereStr}\nGROUP BY ${a1}.${info.id}${info.name ? `, ${a1}.${info.name}` : ""}\nORDER BY total_value DESC${limitStr};`;
        }

        return `SELECT ${a1}.${info.id}${info.name ? `, ${a1}.${info.name}` : ""}${secInfo.num ? `, ${a2}.${secInfo.num}` : ""}\nFROM ${mainTbl.name} ${a1}\nJOIN ${secTbl.name} ${a2} ON ${a1}.${joinCol} = ${a2}.${joinCol}${whereStr}${limitStr};`;
    }

    // Render Results
    function renderResult(sql, latency, tokens, version) {
        sqlOutput.textContent = sql;
        latencyVal.textContent = `${latency} ms`;
        tokensVal.textContent = `${tokens}`;
        versionVal.textContent = version;
    }

    // Update Table Preview
    function updateTablePreview(key) {
        const mock = MOCK_RESULTS[key] || MOCK_RESULTS.ecommerce;
        
        let html = `<thead><tr>${mock.headers.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>`;
        mock.rows.forEach(row => {
            html += `<tr>${row.map(cell => `<td>${cell}</td>`).join("")}</tr>`;
        });
        html += `</tbody>`;

        previewTable.innerHTML = html;
    }

    // Copy Button
    copyBtn.addEventListener("click", () => {
        const text = sqlOutput.textContent;
        if (text) {
            navigator.clipboard.writeText(text);
            copyBtn.innerHTML = `<i class="fa-solid fa-check"></i>`;
            setTimeout(() => {
                copyBtn.innerHTML = `<i class="fa-regular fa-copy"></i>`;
            }, 2000);
        }
    });

    // Explain Button
    explainBtn.addEventListener("click", () => {
        alert("SQL Logic Explanation:\n\n1. JOINs the primary entities specified in schema DDL.\n2. Filters results based on natural language constraints.\n3. Applies aggregation (SUM, COUNT) and orders output deterministically.");
    });
});
