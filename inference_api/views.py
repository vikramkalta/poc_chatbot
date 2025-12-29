from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os
from test_inference import infer, default_weights_name

@csrf_exempt
def inference_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST request required."}, status=405)
    try:
        data = json.loads(request.body.decode())
        model_choice = data.get("model_choice", "gpt2-medium (355M)")
        weights = data.get("weights") or default_weights_name(model_choice)
        instruction = data.get(
            "instruction", "Provide guidance for the following user query."
        )
        input_field = data.get("input", "my transfer is still showing pending")
        max_new_tokens = int(data.get("max_new_tokens", 256))
        if not os.path.exists(weights):
            return JsonResponse({"error": f"Weights not found: {weights}"}, status=400)
        response = infer(
            weights,
            model_choice,
            instruction,
            input_field,
            max_new_tokens,
            return_response=True,
        )
        return JsonResponse({"response": response})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)