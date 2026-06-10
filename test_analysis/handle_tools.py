import json
from .disease_models import predict_hypertension_risk
from .scrapers import scrape_doctors
def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    function_name = tool_call.function.name
    if function_name == 'predict_hypertension_risk':
        args = json.loads(tool_call.function.arguments)
        content = predict_hypertension_risk(**args)
        return {"role": "tool", "content": content, "tool_call_id": tool_call.id}




def handle_doctors_call(message):
    tool_call = message.tool_calls[0]
    function_name = tool_call.function.name
    if function_name == 'scrape_doctors':
        args = json.loads(tool_call.function.arguments)
        content = scrape_doctors(city=args.get('city'),
                                 speciality=args.get('speciality'),
                                 region=args.get('region'),
                                 insurance=args.get('insurance'))
        return {"role": "tool", "content": content, "tool_call_id": tool_call.id}
