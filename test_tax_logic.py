import asyncio
from backend.services.report_generator import ReportGenerator

def test_tax_logic():
    report_gen = ReportGenerator()
    transactions = [
        {"coin_id": "bitcoin", "type": "BUY", "quantity": 1.0, "price": 40000, "timestamp": "2024-01-01"},
        {"coin_id": "bitcoin", "type": "SELL", "quantity": 0.5, "price": 50000, "timestamp": "2024-02-01"}
    ]
    market_data_map = {"bitcoin": 60000}
    try:
        report = report_gen.generate_tax_report(transactions, market_data_map)
        print("Success!", report.keys())
        print("Summary:", report.get("summary"))
    except Exception as e:
        import traceback
        traceback.print_exc()

test_tax_logic()
