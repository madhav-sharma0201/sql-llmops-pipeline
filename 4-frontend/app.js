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

    // Smart Mock SQL Generator Logic (Client-side fallback)
    function generateMockSql(presetKey, question) {
        const qLower = question.toLowerCase();

        // 1. Anti-Join Returns Query (Never returned any product in 2024)
        if (qLower.includes("never returned") || (qLower.includes("returned") && (qLower.includes("not") || qLower.includes("never")))) {
            return `SELECT c.customer_id, c.name\nFROM customers c\nJOIN orders o ON c.customer_id = o.customer_id\nWHERE o.status = 'completed' AND strftime('%Y', o.order_date) = '2024'\n  AND c.customer_id NOT IN (\n    SELECT DISTINCT o2.customer_id\n    FROM orders o2\n    JOIN returns r ON o2.order_id = r.order_id\n    WHERE o2.status = 'completed' AND strftime('%Y', o2.order_date) = '2024'\n  );`;
        }
        
        // 2. Discount & Multi-table Spend Queries
        if (qLower.includes("top 5") || qLower.includes("spend") || qLower.includes("discount")) {
            if (qLower.includes("discount")) {
                return `SELECT c.customer_id, c.name, SUM(oi.quantity * oi.unit_price * (1 - oi.discount_percent / 100.0)) AS total_spend, COUNT(DISTINCT o.order_id) AS distinct_orders\nFROM customers c\nJOIN orders o ON c.customer_id = o.customer_id\nJOIN order_items oi ON o.order_id = oi.order_id\nWHERE o.status = 'completed' AND strftime('%Y', o.order_date) = '2024'\nGROUP BY c.customer_id, c.name\nORDER BY total_spend DESC, c.customer_id ASC\nLIMIT 5;`;
            }
            return `SELECT c.customer_id, c.name AS customer_name, SUM(o.total_amount) AS total_spend, COUNT(o.order_id) AS order_count\nFROM customers c\nJOIN orders o ON c.customer_id = o.customer_id\nWHERE o.status = 'completed' AND strftime('%Y', o.order_date) = '2024'\nGROUP BY c.customer_id, c.name\nORDER BY total_spend DESC\nLIMIT 5;`;
        }
        
        // 3. HR & Salary Queries
        if (qLower.includes("engineering") || qLower.includes("salary")) {
            return `SELECT e.emp_id, e.first_name || ' ' || e.last_name AS full_name, d.dept_name AS department, e.salary, e.hire_date\nFROM employees e\nJOIN departments d ON e.dept_id = d.dept_id\nWHERE d.dept_name = 'Engineering' AND e.salary > 80000\nORDER BY e.hire_date ASC;`;
        }

        // 4. SaaS MRR Queries
        if (qLower.includes("mrr") || qLower.includes("recurring")) {
            return `SELECT plan_tier, COUNT(sub_id) AS active_subscriptions, SUM(mrr_amount) AS total_mrr\nFROM subscriptions\nWHERE status = 'active'\nGROUP BY plan_tier\nORDER BY total_mrr DESC;`;
        }

        // 5. Default Clean Fallback
        return `SELECT customer_id, name\nFROM customers\nWHERE account_status = 'active'\nLIMIT 10;`;
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
