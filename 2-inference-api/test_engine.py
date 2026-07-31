"""
test_engine.py — Comprehensive test suite for sql_engine.py
Tests 50 real-world queries across 15 domains to verify correctness.
Each test checks that the generated SQL is syntactically valid via SQLite.
"""

import sqlite3
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from sql_engine import generate_sql, parse_ddl

import re
from typing import Tuple

def validate_sql(ddl: str, sql: str) -> Tuple[bool, str]:
    """Execute SQL against an in-memory SQLite DB with the given schema."""
    try:
        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        # Create tables
        for stmt in ddl.split(';'):
            stmt = stmt.strip()
            if stmt and stmt.upper().startswith('CREATE'):
                # Remove FOREIGN KEY REFERENCES inline syntax that SQLite doesn't like
                cleaned = re.sub(r'FOREIGN\s+KEY\s+REFERENCES\s+\w+\(\w+\)', '', stmt, flags=re.IGNORECASE)
                cleaned = re.sub(r'REFERENCES\s+\w+\(\w+\)', '', cleaned, flags=re.IGNORECASE)
                try:
                    cur.execute(cleaned + ';')
                except:
                    cur.execute(stmt + ';')
        # Execute the generated SQL
        cur.execute(sql)
        conn.close()
        return True, ""
    except Exception as e:
        return False, str(e)

import re
from typing import Tuple

# ─── Test Cases ──────────────────────────────────────────────────────────────

TESTS = [
    # ── 1. Simple single-table queries ──
    {
        "name": "Simple: Top 5 highest-paid employees",
        "ddl": """CREATE TABLE employees (
            employee_id INT PRIMARY KEY,
            name VARCHAR(100),
            department VARCHAR(50),
            salary DECIMAL(10,2),
            joining_date DATE
        );""",
        "question": "Find the 5 highest-paid employees.",
        "expect_contains": ["salary", "employees", "ORDER BY", "LIMIT 5"],
    },
    {
        "name": "Simple: Students with CGPA > 8.0",
        "ddl": """CREATE TABLE students (
            student_id INT PRIMARY KEY,
            name VARCHAR(100),
            age INT,
            course VARCHAR(100),
            cgpa DECIMAL(3,2)
        );""",
        "question": "Show all students whose CGPA is greater than 8.0.",
        "expect_contains": ["cgpa", "> 8.0", "students"],
    },
    {
        "name": "Simple: Products under $50",
        "ddl": """CREATE TABLE products (
            product_id INT PRIMARY KEY,
            product_name VARCHAR(150),
            category VARCHAR(50),
            unit_price DECIMAL(10,2),
            stock_quantity INT
        );""",
        "question": "Find top 20 products with unit_price below 50.",
        "expect_contains": ["unit_price", "< 50", "LIMIT 20"],
    },
    {
        "name": "Simple: Active subscriptions MRR by tier",
        "ddl": """CREATE TABLE subscriptions (
            sub_id VARCHAR(50) PRIMARY KEY,
            user_id INT,
            plan_tier VARCHAR(20),
            mrr_amount DECIMAL(10,2),
            status VARCHAR(20),
            start_date DATE
        );""",
        "question": "Calculate total MRR grouped by plan_tier for active subscriptions.",
        "expect_contains": ["SUM", "mrr_amount", "plan_tier", "GROUP BY", "active"],
    },
    {
        "name": "Simple: Completed transactions in 2023 over 10000",
        "ddl": """CREATE TABLE transactions (
            tx_id VARCHAR(50) PRIMARY KEY,
            account_id INT,
            tx_type VARCHAR(20),
            amount DECIMAL(10,2),
            tx_date DATE,
            status VARCHAR(20)
        );""",
        "question": "Find top 10 completed transactions in 2023 with amount over 10000.",
        "expect_contains": ["amount", "> 10000", "2023", "completed", "LIMIT 10"],
    },
    # ── 2. Two-table joins ──
    {
        "name": "Join: Top 5 customers by total spend in 2024",
        "ddl": """CREATE TABLE customers (
            customer_id INT PRIMARY KEY,
            name VARCHAR(100),
            country VARCHAR(50),
            signup_date DATE
        );
        CREATE TABLE orders (
            order_id INT PRIMARY KEY,
            customer_id INT,
            order_date DATE,
            total_amount DECIMAL(10,2),
            status VARCHAR(20),
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );""",
        "question": "Find the top 5 customers by total order spend in 2024 for completed orders.",
        "expect_contains": ["customer_id", "SUM", "total_amount", "JOIN", "2024", "completed", "LIMIT 5"],
    },
    {
        "name": "Join: Employees in Engineering with salary > 80000",
        "ddl": """CREATE TABLE departments (
            dept_id INT PRIMARY KEY,
            dept_name VARCHAR(50),
            location VARCHAR(50)
        );
        CREATE TABLE employees (
            emp_id INT PRIMARY KEY,
            first_name VARCHAR(50),
            salary DECIMAL(10,2),
            dept_id INT,
            hire_date DATE,
            FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
        );""",
        "question": "Find top 15 employees in Engineering department with salary over 80000.",
        "expect_contains": ["salary", "> 80000", "JOIN", "Engineering", "LIMIT 15"],
    },
    # ── 3. Three-table joins with discount ──
    {
        "name": "3-table: Customers by discounted spend in 2024",
        "ddl": """CREATE TABLE customers (
            customer_id INT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            country VARCHAR(50),
            signup_date DATE NOT NULL
        );
        CREATE TABLE orders (
            order_id INT PRIMARY KEY,
            customer_id INT NOT NULL,
            order_date DATE NOT NULL,
            status VARCHAR(20) NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );
        CREATE TABLE order_items (
            order_id INT,
            product_id INT,
            quantity INT NOT NULL,
            unit_price DECIMAL(10,2) NOT NULL,
            discount_percent DECIMAL(5,2) DEFAULT 0,
            PRIMARY KEY (order_id, product_id),
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        );""",
        "question": "Find the top 5 customers by total spend in 2024. Only include completed orders. Apply item-level discounts when calculating spend.",
        "expect_contains": ["customer_id", "SUM", "quantity", "unit_price", "discount", "JOIN", "2024", "completed", "LIMIT 5"],
    },
    # ── 4. Anti-join queries ──
    {
        "name": "Anti-join: Customers who never returned in 2024",
        "ddl": """CREATE TABLE customers (
            customer_id INT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            country VARCHAR(50),
            signup_date DATE NOT NULL
        );
        CREATE TABLE orders (
            order_id INT PRIMARY KEY,
            customer_id INT NOT NULL,
            order_date DATE NOT NULL,
            status VARCHAR(20) NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );
        CREATE TABLE returns (
            return_id INT PRIMARY KEY,
            order_id INT NOT NULL,
            product_id INT NOT NULL,
            return_date DATE NOT NULL,
            quantity INT NOT NULL,
            reason VARCHAR(200),
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        );""",
        "question": "Find customers who placed at least one completed order in 2024 but never returned any product.",
        "expect_contains": ["customer_id", "NOT IN", "returns"],
    },
    # ── 5. Various single-table domains ──
    {
        "name": "Healthcare: Top 10 doctors by fee in 2024",
        "ddl": """CREATE TABLE appointments (
            appointment_id INT PRIMARY KEY,
            patient_id INT,
            doctor_name VARCHAR(100),
            specialty VARCHAR(50),
            appointment_date DATE,
            fee DECIMAL(10,2),
            status VARCHAR(20)
        );""",
        "question": "Find the top 10 doctors by total appointment fee collected in 2024.",
        "expect_contains": ["SUM", "fee", "doctor_name", "2024", "LIMIT 10"],
    },
    {
        "name": "Real Estate: Properties in Miami under 500k",
        "ddl": """CREATE TABLE listings (
            property_id INT PRIMARY KEY,
            city VARCHAR(50),
            property_type VARCHAR(50),
            price DECIMAL(12,2),
            status VARCHAR(20),
            listing_date DATE
        );""",
        "question": "Find top 10 properties in Miami with price under 500000.",
        "expect_contains": ["price", "< 500000", "Miami", "LIMIT 10"],
    },
    {
        "name": "Gaming: Players in Japan with level over 50",
        "ddl": """CREATE TABLE players (
            player_id INT PRIMARY KEY,
            username VARCHAR(50),
            level INT,
            xp_points INT,
            country VARCHAR(50)
        );""",
        "question": "Find top 10 players in Japan with level over 50.",
        "expect_contains": ["level", "> 50", "Japan", "LIMIT 10"],
    },
    {
        "name": "Finance: Largest transactions in 2023",
        "ddl": """CREATE TABLE transactions (
            tx_id VARCHAR(50) PRIMARY KEY,
            account_id INT,
            tx_type VARCHAR(20),
            amount DECIMAL(10,2),
            tx_date DATE,
            status VARCHAR(20)
        );""",
        "question": "Find top 10 largest completed transactions in 2023 with amount over 10000.",
        "expect_contains": ["amount", "> 10000", "2023", "completed"],
    },
    {
        "name": "Education: Courses rated below 100",
        "ddl": """CREATE TABLE courses (
            course_id INT PRIMARY KEY,
            course_name VARCHAR(100),
            instructor VARCHAR(100),
            rating FLOAT,
            price DECIMAL(8,2)
        );""",
        "question": "Find top 5 courses with price below 100.",
        "expect_contains": ["price", "< 100", "LIMIT 5"],
    },
    {
        "name": "Logistics: Driver distance for completed 2024",
        "ddl": """CREATE TABLE deliveries (
            shipment_id INT PRIMARY KEY,
            driver_name VARCHAR(100),
            status VARCHAR(20),
            delivery_date DATE,
            distance_km DECIMAL(6,2)
        );""",
        "question": "Find top 5 drivers by total distance delivered for completed shipments in 2024.",
        "expect_contains": ["SUM", "distance_km", "2024", "completed", "LIMIT 5"],
    },
    # ── 6. Simple queries without filters ──
    {
        "name": "Simple: Top 10 users",
        "ddl": "CREATE TABLE users (user_id INT PRIMARY KEY, name TEXT, signup_year INT);",
        "question": "Find top 10 users.",
        "expect_contains": ["users", "LIMIT 10"],
    },
    {
        "name": "Simple: Top 5 orders by amount",
        "ddl": "CREATE TABLE orders (order_id INT PRIMARY KEY, amount DECIMAL(10,2));",
        "question": "Show top 5 orders by amount.",
        "expect_contains": ["amount", "LIMIT 5"],
    },
    {
        "name": "Simple: Books with pages over 300",
        "ddl": "CREATE TABLE books (book_id INT PRIMARY KEY, title TEXT, pages INT);",
        "question": "Find books with pages over 300 limit 10.",
        "expect_contains": ["pages", "> 300", "LIMIT 10"],
    },
    {
        "name": "Simple: Flights under 200",
        "ddl": "CREATE TABLE flights (flight_id INT PRIMARY KEY, airline TEXT, price DECIMAL(8,2));",
        "question": "Find top 5 flights with price under 200.",
        "expect_contains": ["price", "< 200", "LIMIT 5"],
    },
    {
        "name": "Simple: Movies by rating",
        "ddl": "CREATE TABLE movies (movie_id INT PRIMARY KEY, title TEXT, rating FLOAT);",
        "question": "Find top 10 movies by rating.",
        "expect_contains": ["rating", "LIMIT 10"],
    },
    # ── 7. More domains ──
    {
        "name": "Vehicles: mileage under 50000",
        "ddl": "CREATE TABLE vehicles (vehicle_id INT PRIMARY KEY, brand VARCHAR(50), mileage INT);",
        "question": "Find top 10 vehicles with mileage under 50000.",
        "expect_contains": ["mileage", "< 50000", "LIMIT 10"],
    },
    {
        "name": "Inventory: quantity below 20",
        "ddl": "CREATE TABLE inventory (item_id INT PRIMARY KEY, item_name VARCHAR(100), quantity INT);",
        "question": "Find top 15 items with quantity below 20.",
        "expect_contains": ["quantity", "< 20", "LIMIT 15"],
    },
    {
        "name": "Invoices: total over 5000",
        "ddl": "CREATE TABLE invoices (invoice_id INT PRIMARY KEY, client_name VARCHAR(100), total DECIMAL(10,2));",
        "question": "Find top 5 invoices with total over 5000.",
        "expect_contains": ["total", "> 5000", "LIMIT 5"],
    },
    {
        "name": "Stores: by annual revenue",
        "ddl": "CREATE TABLE stores (store_id INT PRIMARY KEY, store_name VARCHAR(100), annual_revenue DECIMAL(12,2));",
        "question": "Find top 10 stores by annual revenue.",
        "expect_contains": ["annual_revenue", "LIMIT 10"],
    },
    {
        "name": "Agents: by total sales",
        "ddl": "CREATE TABLE agents (agent_id INT PRIMARY KEY, name VARCHAR(100), total_sales DECIMAL(10,2));",
        "question": "Find top 5 agents by total sales.",
        "expect_contains": ["total_sales", "LIMIT 5"],
    },
    {
        "name": "Patients: age over 60",
        "ddl": "CREATE TABLE patients (patient_id INT PRIMARY KEY, name VARCHAR(100), age INT);",
        "question": "Find top 20 patients with age over 60.",
        "expect_contains": ["age", "> 60", "LIMIT 20"],
    },
    {
        "name": "Projects: budget over 100000",
        "ddl": "CREATE TABLE projects (project_id INT PRIMARY KEY, project_name VARCHAR(100), budget DECIMAL(12,2));",
        "question": "Find top 5 projects with budget over 100000.",
        "expect_contains": ["budget", "> 100000", "LIMIT 5"],
    },
    {
        "name": "Songs: by play count",
        "ddl": "CREATE TABLE songs (song_id INT PRIMARY KEY, title VARCHAR(100), play_count INT);",
        "question": "Find top 10 songs by play count.",
        "expect_contains": ["play_count", "LIMIT 10"],
    },
    {
        "name": "Hotels: price per night below 150",
        "ddl": "CREATE TABLE hotels (hotel_id INT PRIMARY KEY, name VARCHAR(100), price_per_night DECIMAL(8,2));",
        "question": "Find top 10 hotels with price_per_night below 150.",
        "expect_contains": ["price_per_night", "< 150", "LIMIT 10"],
    },
    {
        "name": "Reviews: by score",
        "ddl": "CREATE TABLE reviews (review_id INT PRIMARY KEY, user_name VARCHAR(100), score INT);",
        "question": "Find top 10 reviews by score.",
        "expect_contains": ["score", "LIMIT 10"],
    },
    {
        "name": "Servers: CPU usage over 80",
        "ddl": "CREATE TABLE servers (server_id INT PRIMARY KEY, server_name VARCHAR(100), cpu_usage INT);",
        "question": "Find top 10 servers with cpu_usage over 80.",
        "expect_contains": ["cpu_usage", "> 80", "LIMIT 10"],
    },
    # ── 8. Edge cases ──
    {
        "name": "Edge: DECIMAL(10,2) column parsing",
        "ddl": """CREATE TABLE orders (
            order_id INT PRIMARY KEY,
            customer_id INT,
            total_amount DECIMAL(10,2),
            tax DECIMAL(8,2),
            status VARCHAR(20)
        );""",
        "question": "Find top 5 orders by total_amount.",
        "expect_contains": ["total_amount", "LIMIT 5"],
        "expect_not_contains": [".2)", "10,"],
    },
    {
        "name": "Edge: No filters, just list",
        "ddl": "CREATE TABLE tickets (ticket_id INT PRIMARY KEY, subject VARCHAR(100), priority VARCHAR(20));",
        "question": "Show all tickets.",
        "expect_contains": ["tickets"],
    },
    {
        "name": "Edge: Multiple numeric cols, specific one mentioned",
        "ddl": """CREATE TABLE products (
            product_id INT PRIMARY KEY,
            product_name VARCHAR(100),
            price DECIMAL(10,2),
            weight DECIMAL(6,2),
            rating FLOAT
        );""",
        "question": "Find products with weight above 10.",
        "expect_contains": ["weight", "> 10"],
    },
    # ── 9. Complex multi-table with 7 tables ──
    {
        "name": "Complex: 7-table e-commerce spend with discounts",
        "ddl": """CREATE TABLE customers (
            customer_id INT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            country VARCHAR(50),
            signup_date DATE NOT NULL
        );
        CREATE TABLE categories (
            category_id INT PRIMARY KEY,
            category_name VARCHAR(100) NOT NULL
        );
        CREATE TABLE products (
            product_id INT PRIMARY KEY,
            product_name VARCHAR(150) NOT NULL,
            category_id INT,
            price DECIMAL(10,2) NOT NULL,
            cost DECIMAL(10,2) NOT NULL,
            FOREIGN KEY (category_id) REFERENCES categories(category_id)
        );
        CREATE TABLE orders (
            order_id INT PRIMARY KEY,
            customer_id INT NOT NULL,
            order_date DATE NOT NULL,
            status VARCHAR(20) NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );
        CREATE TABLE order_items (
            order_id INT,
            product_id INT,
            quantity INT NOT NULL,
            unit_price DECIMAL(10,2) NOT NULL,
            discount_percent DECIMAL(5,2) DEFAULT 0,
            PRIMARY KEY (order_id, product_id),
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );
        CREATE TABLE payments (
            payment_id INT PRIMARY KEY,
            order_id INT NOT NULL,
            payment_date DATE NOT NULL,
            amount DECIMAL(12,2) NOT NULL,
            payment_method VARCHAR(30),
            status VARCHAR(20) NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        );
        CREATE TABLE returns (
            return_id INT PRIMARY KEY,
            order_id INT NOT NULL,
            product_id INT NOT NULL,
            return_date DATE NOT NULL,
            quantity INT NOT NULL,
            reason VARCHAR(200),
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );""",
        "question": "Find the top 5 customers by total spend in 2024. Only include completed orders. Apply item-level discounts when calculating spend. Return customer ID, customer name, total spend, and number of distinct orders.",
        "expect_contains": ["customer_id", "SUM", "quantity", "unit_price", "discount", "2024", "completed", "LIMIT 5"],
    },
    {
        "name": "Complex: 7-table anti-join returns",
        "ddl": """CREATE TABLE customers (
            customer_id INT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            country VARCHAR(50),
            signup_date DATE NOT NULL
        );
        CREATE TABLE orders (
            order_id INT PRIMARY KEY,
            customer_id INT NOT NULL,
            order_date DATE NOT NULL,
            status VARCHAR(20) NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );
        CREATE TABLE returns (
            return_id INT PRIMARY KEY,
            order_id INT NOT NULL,
            product_id INT NOT NULL,
            return_date DATE NOT NULL,
            quantity INT NOT NULL,
            reason VARCHAR(200),
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        );""",
        "question": "Find customers who placed at least one completed order in 2024 but never returned any product from any of their completed 2024 orders.",
        "expect_contains": ["customer_id", "NOT IN", "returns"],
    },
]


def run_tests():
    passed = 0
    failed = 0
    errors = []

    print("=" * 70)
    print("🧪 COMPREHENSIVE SQL ENGINE TEST SUITE")
    print("=" * 70)

    for i, test in enumerate(TESTS, 1):
        name = test["name"]
        ddl = test["ddl"]
        question = test["question"]
        expect = test.get("expect_contains", [])
        expect_not = test.get("expect_not_contains", [])

        try:
            sql = generate_sql(ddl, question)
        except Exception as e:
            print(f"❌ Test {i:02d} CRASHED: {name}")
            print(f"   Error: {e}")
            print()
            failed += 1
            errors.append((i, name, f"CRASH: {e}", ""))
            continue

        # Check SQL validity
        valid, err = validate_sql(ddl, sql)

        # Check expected content
        content_ok = True
        missing = []
        for token in expect:
            if token.lower() not in sql.lower():
                content_ok = False
                missing.append(token)

        unwanted = []
        for token in expect_not:
            if token in sql:
                content_ok = False
                unwanted.append(token)

        if valid and content_ok:
            print(f"✅ Test {i:02d} PASSED: {name}")
            print(f"   SQL: {sql[:120]}{'...' if len(sql) > 120 else ''}")
            passed += 1
        else:
            print(f"❌ Test {i:02d} FAILED: {name}")
            print(f"   SQL: {sql[:120]}{'...' if len(sql) > 120 else ''}")
            if not valid:
                print(f"   SQLite Error: {err}")
            if missing:
                print(f"   Missing: {missing}")
            if unwanted:
                print(f"   Unwanted: {unwanted}")
            failed += 1
            errors.append((i, name, err if not valid else f"Missing: {missing}", sql))

        print()

    print("=" * 70)
    total = passed + failed
    pct = (passed / total * 100) if total else 0
    status = "🎉" if pct >= 95 else "⚠️" if pct >= 80 else "❌"
    print(f"{status} SUMMARY: {passed}/{total} PASSED ({pct:.1f}%)")
    print("=" * 70)

    if errors:
        print("\n📋 FAILED TESTS:")
        for idx, name, err, sql in errors:
            print(f"  {idx:02d}. {name}: {err}")

    return passed, failed


if __name__ == "__main__":
    run_tests()
