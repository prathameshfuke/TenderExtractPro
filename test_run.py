import sys
import traceback
import faulthandler
faulthandler.enable()

try:
    from tender_extraction.main import TenderExtractionPipeline
    pipeline = TenderExtractionPipeline()
    result = pipeline.run('dataset/globaltender1576.pdf', 'outputs/test_run.json')
    print("SUCCESS")
    print(f"  Specs: {len(result.get('technical_specifications', []))}")
    print(f"  Deliverables: {len(result.get('scope_of_work', {}).get('deliverables', []))}")
    print(f"  Accuracy: {result.get('accuracy_score', 0):.1f}%")
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
