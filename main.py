import os
import json
import base64
import asyncio
import websockets
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream
from dotenv import load_dotenv
from datetime import datetime
import re
from twilio.rest import Client
from flask import render_template
from gtts import gTTS
import io
from pydub import AudioSegment
import re
import subprocess
import tempfile
from fastapi.responses import RedirectResponse
from fastapi import Form, HTTPException
from fastapi.staticfiles import StaticFiles

def contains_chinese(text):
    chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]+')
    return bool(chinese_char_pattern.search(text))

load_dotenv()

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_phone_number = os.getenv("TWILIO_PHONE_NUMBER")
twilio_client = Client(account_sid, auth_token)
stream_id_to_call_id_map = dict()

# Switch of using gtts instead of using openai's media stream output
# Because openai's media stream output has a lot of package lost sometimes
use_tts = True

# Load the JSON menu
with open("menus/dragon_chinese_restaurant_menu.json", "r", encoding="utf-8") as f:
    menu_data = json.load(f)


def normalize_string(s):
    """Remove punctuation and convert to lowercase."""
    return re.sub(r'[^\w\s]', '', s.lower())

def calculate_similarity(s1, s2):
    """Calculate the similarity between two strings."""
    s1 = normalize_string(s1)
    s2 = normalize_string(s2)
    s1_words = set(s1.split())
    s2_words = set(s2.split())
    return len(s1_words & s2_words) / len(s1_words | s2_words)

def check_menu_items(items):
    """
    Check if the ordered items are in the menu.
    Returns a JSON string with information about each item.
    """
    results = []
    for item in items:
        found = False
        closest_match = None
        highest_similarity = 0
        item_info = None

        normalized_item = normalize_string(item)

        for section in menu_data['menu']:
            for menu_item in section['items']:
                normalized_menu_item = normalize_string(menu_item['name'])
                similarity = calculate_similarity(normalized_item, normalized_menu_item)

                if similarity == 1.0 or normalized_item in normalized_menu_item:  # Exact match or substring
                    found = True
                    item_info = menu_item
                    break
                elif similarity > highest_similarity:
                    highest_similarity = similarity
                    closest_match = menu_item

            if found:
                break

        if found:
            results.append({
                "item": item,
                "status": "found",
                "info": item_info
            })
        elif highest_similarity > 0.5:  # Lowered threshold for similarity
            results.append({
                "item": item,
                "status": "close_match",
                "closest_match": closest_match['name'],
                "info": closest_match
            })
        else:
            results.append({
                "item": item,
                "status": "not_found"
            })

    print(f"Debug - check_menu_items result: {results}")
    return json.dumps(results)  # Return as JSON string

def transform_order(order):
    tax_rate = menu_data['tax_rate']
    transformed_order = {"items": [], "subtotal": 0, "tax": 0, "total": 0}

    # Process each item in the order
    for item in order["items"]:
        item_id = item.get("id")
        sub_id = item.get("sub_id")  # This could be None or a specific sub_id/size
        quantity = item.get("quantity", 1)
        
        # Initialize variables to hold item details if found
        menu_item = None
        price_per_unit = None
        item_name = None

        # Find the item in menu_data by searching each section
        for section in menu_data['menu']:
            for menu in section['items']:
                # Match both id and sub_id (if present)
                if menu.get("id") == item_id and menu.get("sub_id") == sub_id:
                    item_name = menu["name"]
                    price_per_unit = menu["price"]
                    menu_item = menu
                    break
                elif menu.get("id") == item_id and sub_id is None and not isinstance(menu.get("price"), dict):
                    # Handle items without sub_id, such as those with a single price
                    item_name = menu["name"]
                    price_per_unit = menu["price"]
                    menu_item = menu
                    break
            if menu_item:
                break

        # If the item is found in the menu, add it to the transformed order
        if menu_item and price_per_unit is not None:
            item_total_price = price_per_unit * quantity
            transformed_order["items"].append({
                "item_id": item_id,
                "sub_id": sub_id,
                "item_name_english": item_name,
                "quantity": quantity,
                "price_per_unit": price_per_unit,
                "item_total_price": item_total_price
            })
            transformed_order["subtotal"] += item_total_price
        else:
            print(f"Warning: Item with id {item_id} and sub_id {sub_id} not found in menu.")

    # Calculate tax and total
    transformed_order["tax"] = round(transformed_order["subtotal"] * tax_rate, 5)
    transformed_order["total"] = round(transformed_order["subtotal"] + transformed_order["tax"], 2)
    
    return transformed_order

def process_order(order, stream_sid):
    """
    Process the order and save it to a JSON file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    order_id = f"order_{timestamp}"
    os.makedirs('./orders', exist_ok=True)
    filename = f"./orders/order_{timestamp}.json"
    transformed_order = transform_order(order)
    
    with open(filename, 'w') as f:
        json.dump(transformed_order, f, indent=4)


    call_sid = stream_id_to_call_id_map[stream_sid]
    call = twilio_client.calls(call_sid).fetch()
    caller_number = call._from
    print(f"Caller Number: {caller_number}")

    # Map order_id to caller_number and save the mapping
    order_caller_mapping[order_id] = caller_number
    save_order_caller_mapping(order_caller_mapping)

    message = twilio_client.messages.create(
        body=f"ORDER CONFIRMATION FROM DRAGON RESTAURANT:\nHello, your order has been successfully placed. You can view the details of your order and complete your payment here: https://ruofan-test-1.ngrok.app/pay/{order_id}",
        from_=twilio_phone_number,
        to=caller_number       # Customer's phone number
    )

    print(f"Order saved to {filename}")
    return json.dumps({"message": f"Order processed and saved to {filename}"})  # Return JSON string

TOOLS = [
    {
        "type": "function",
        "name": "check_menu_items",
        "description": "Call this function EVERY TIME a customer mentions ANY item they want to order, even if the same item was checked before. This includes when they're adding items, or changing quantities. If items are mentioned in a language other than English, first convert them to English before passing them to this function. Example: If customer says '炒面' or 'chow mein', pass 'chow mein' to the function.",
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of item names in English. Convert any non-English item names to their English equivalents before adding to this list."
                }
            },
            "required": ["items"]
        }
    },
    {
        "type": "function",
        "name": "process_order",
        "description": "Processes customer order and returns the order details. Only call this function after the customer has explicitly agreed that the order is correct. Do not call this function during the ordering process or before final confirmation.",
        "parameters": {
            "type": "object",
            "properties": {
                "order": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "integer"},
                                    "sub_id": {
                                        "type": ["integer", "null"],
                                        "description": "Sub-id of the item if applicable, null otherwise."
                                    },
                                    "quantity": {"type": "integer"},
                                    "size": {
                                        "type": "string",
                                        "enum": ["Small", "Large", "Shallow", "Medium", "None"],
                                        "description": "The size of the item. If the item has no sizes, the size field should remain None."
                                    },
                                    "choice": {
                                        "type": "string",
                                        "enum": ["Chicken", "Beef", "Shrimp", "None"],
                                        "description": "The choice of the item. If the item has no choices, the choice field should remain None."
                                    },
                                },
                                "required": ["id", "sub_id", "quantity", "size", "choice"],
                                "additionalProperties": False
                            }
                        },
                    },
                    "required": ["items"],
                    "additionalProperties": False
                }
            },
            "required": ["order"]
        }
    }
]

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') # requires OpenAI Realtime API Access
PORT = int(os.getenv('PORT', 5050))
SYSTEM_MESSAGE = (
f"""
You are {menu_data['restaurant_name']}'s order assistant. Be direct and only respond to what customers ask. Follow these rules:

The menu data is here {menu_data}

1. **Core Behaviors**:
    - Answer only what is asked
    - Speak naturally without special characters
    - Keep responses brief
    - Only ask follow-up questions when necessary for order accuracy
    - when user asks for recommendations, recommend 3 to 4 items will be good, do not recommend more than 3 to 4 items.

2. **Order Taking**:
    - For unavailable items: Simply state it's not available and ask for an alternative
    - For items with options: Only ask for size/choice if customer hasn't specified
    - After each item: Only ask "Would you like anything else?"
    - Before processing the order you must confirm the order. Final confirmation format: "You've ordered [items], subtotal [amount], total with {menu_data['tax_rate'] * 100}% tax is [total]. Is this correct?"

3. **Non-Order Requests**:
    - For business inquiries: "I'll forward your request to the manager."
"""
)
VOICE = 'shimmer'
LOG_EVENT_TYPES = [
    'response.content.done', 'rate_limits.updated', 'response.done',
    'input_audio_buffer.committed', 'input_audio_buffer.speech_stopped',
    'input_audio_buffer.speech_started', 'session.created'
]

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# File to track paid orders
PAID_ORDERS_FILE = "./orders/paid_orders.json"

# Load paid orders from file
def load_paid_orders():
    if os.path.exists(PAID_ORDERS_FILE):
        with open(PAID_ORDERS_FILE, "r") as file:
            return json.load(file)
    return {}

# Save paid orders to file
def save_paid_orders(paid_orders):
    with open(PAID_ORDERS_FILE, "w") as file:
        json.dump(paid_orders, file, indent=4)

# Initialize paid orders
paid_orders = load_paid_orders()

@app.get("/pay/{order_id}", response_class=HTMLResponse)
async def payment_page(order_id: str, request: Request):
    """
    Endpoint to display the payment page with the order summary included.
    """
    order_dir = "./orders"
    order_filepath = os.path.join(order_dir, order_id + ".json")
    
    if os.path.exists(order_filepath):
        with open(order_filepath, "r") as file:
            order_data = json.load(file)
        
        if order_id in paid_orders:
            # Render a friendly "Already Paid" page
            return templates.TemplateResponse(
                "already_paid.html",
                {
                    "request": request,
                    "order_id": order_id,
                    "order_data": order_data,
                }
            )
        
        # Render the payment page with the order summary
        return templates.TemplateResponse(
            "payment_template.html",
            {
                "request": request,
                "order_id": order_id,
                "order_data": order_data,
                "order_total": order_data["total"],
            }
        )
    else:
        raise HTTPException(status_code=404, detail="Order not found")

ORDER_CALLER_MAPPING_FILE = "./orders/order_caller_mapping.json"

# Load the order-to-caller mapping from file
def load_order_caller_mapping():
    if os.path.exists(ORDER_CALLER_MAPPING_FILE):
        with open(ORDER_CALLER_MAPPING_FILE, "r") as file:
            return json.load(file)
    return {}

# Save the order-to-caller mapping to file
def save_order_caller_mapping(mapping):
    with open(ORDER_CALLER_MAPPING_FILE, "w") as file:
        json.dump(mapping, file, indent=4)

# Initialize the order-to-caller mapping
order_caller_mapping = load_order_caller_mapping()

@app.post("/pay/{order_id}")
async def process_payment(order_id: str, card_number: str = Form(...), expiry_date: str = Form(...), cvv: str = Form(...)):
    order_dir = "./orders"
    order_filepath = os.path.join(order_dir, order_id + ".json")
    
    if os.path.exists(order_filepath):
        if order_id in paid_orders:
            raise HTTPException(status_code=400, detail="Order has already been paid.")
        
        # Mock payment processing
        paid_orders[order_id] = True
        save_paid_orders(paid_orders)

        # Retrieve the caller number for the given order_id
        caller_number = order_caller_mapping.get(order_id)
        if caller_number:
            # Send Twilio notification
            message = twilio_client.messages.create(
                body=f"ORDER UPDATE:\nThank you for your payment! Your order (ID: {order_id}) is now being prepared. It will be ready for pickup in approximately 15 minutes.\nWe look forward to serving you!",
                from_=twilio_phone_number,
                to=caller_number       # Customer's phone number
            )
            print(f"Twilio message sent: {message.sid}")

        return RedirectResponse(url=f"/payment-success/{order_id}", status_code=303)
    else:
        raise HTTPException(status_code=404, detail="Order not found.")

@app.get("/payment-success/{order_id}", response_class=HTMLResponse)
async def payment_success(order_id: str, request: Request):
    if order_id not in paid_orders:
        raise HTTPException(status_code=400, detail="Payment not completed for this order.")
    
    return templates.TemplateResponse(
        "payment_success.html",
        {"request": request, "order_id": order_id}
    )



if not OPENAI_API_KEY:
    raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')

@app.get("/", response_class=HTMLResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}

@app.get("/{order_id}")
async def get_order(order_id: str, request: Request):
    order_dir = "./orders"
    order_filepath = os.path.join(order_dir, order_id + ".json")
    
    print("Constructed order_filepath is:", order_filepath)
    if os.path.exists(order_filepath):
        with open(order_filepath, 'r') as file:
            order_data = json.load(file)
            print("Loaded order_data:", order_data)
        
        # Render the order using the external HTML template
        return templates.TemplateResponse(
            "order_template.html", 
            {"request": request, "order_id": order_id, "order_data": order_data}
        )
    else:
        print(f"Order file '{order_filepath}' not found.")
        raise HTTPException(status_code=404, detail="Order not found")

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    response = VoiceResponse()
    # <Say> punctuation to improve text-to-speech flow
    response.say("Welcome to dragon restaurant, how can I help you?")
    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f'wss://{host}/media-stream')
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and OpenAI."""
    print("Client connected")
    await websocket.accept()

    async with websockets.connect(
        'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01',
        extra_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
    ) as openai_ws:
        await send_session_update(openai_ws)
        stream_sid = None

        async def receive_from_twilio():
            """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
            nonlocal stream_sid
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data['event'] == 'media' and openai_ws.open:
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": data['media']['payload']
                        }

                        await openai_ws.send(json.dumps(audio_append))
                    elif data['event'] == 'start':
                        stream_sid = data['start']['streamSid']
                        stream_id_to_call_id_map[stream_sid] = data['start']['callSid']
                        print(f"Incoming stream has started {stream_sid}")
            except WebSocketDisconnect:
                print("Client disconnected.")
                if openai_ws.open:
                    await openai_ws.close()

        async def send_to_twilio():
            """Receive events from the OpenAI Realtime API, send audio back to Twilio."""
            nonlocal stream_sid
            audio_delta = None
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)
                    # Print the type and contents of the response for debugging
                    print(f"Received response type: {response['type']}")
                    print("Response contents:")
                    print(json.dumps(response, indent=4))
                    if response['type'] in LOG_EVENT_TYPES:
                        print(f"Received event: {response['type']}", response)
                    if response['type'] == 'session.updated':
                        print("Session updated successfully:", response)
                    if response['type'] == 'response.audio.delta' and response.get('delta'):
                        if (use_tts):
                            continue
                        # Audio from OpenAI
                        try:
                            audio_payload = base64.b64encode(base64.b64decode(response['delta'])).decode('utf-8')
                            audio_delta = {
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {
                                    "payload": audio_payload
                                }
                            }
                            await websocket.send_json(audio_delta)
                        except Exception as e:
                            print(f"Error processing audio data: {e}")
                    if response['type'] == 'response.done':
                        if (use_tts):
                            try:
                                transcript = response["response"]["output"][0]["content"][0]["transcript"]
                                language = "en"
                                if (contains_chinese(transcript)):
                                    language = "zh"

                                tts = gTTS(text=transcript, lang=language)
                                audio_data = io.BytesIO()
                                tts.write_to_fp(audio_data)
                                audio_data.seek(0)

                                audio_segment = AudioSegment.from_file(audio_data, format="mp3")
                                audio_segment = audio_segment.set_frame_rate(8000).set_channels(1).set_sample_width(2)
                                audio_segment = audio_segment.speedup(playback_speed=1.25)

                                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as pcm_wav_file:
                                    audio_segment.export(pcm_wav_file.name, format="wav")
                                    pcm_wav_file_path = pcm_wav_file.name

                                # Create a second temporary file for mu-law output
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as mulaw_wav_file:
                                    mulaw_wav_file_path = mulaw_wav_file.name

                                subprocess.run([
                                    "ffmpeg", "-y", "-i", pcm_wav_file_path, "-ar", "8000", "-ac", "1", "-f", "mulaw", mulaw_wav_file_path
                                ], stderr=subprocess.PIPE)

                                with open(mulaw_wav_file_path, "rb") as f:
                                    audio_payload = base64.b64encode(f.read()).decode('utf-8')

                                os.remove(pcm_wav_file_path)
                                os.remove(mulaw_wav_file_path)

                                audio_delta = {
                                    "event": "media",
                                    "streamSid": stream_sid,
                                    "media": {
                                        "payload": audio_payload
                                    }
                                }
                                await websocket.send_json(audio_delta)
                            except Exception as e:
                                print("No audio data in this response.done")

                        # Check if there's any output in the response
                        output = response.get('response', {}).get('output', [])
                        for item in output:
                            if item['type'] == 'function_call':
                                try:
                                    arguments = json.loads(item['arguments'])
                                    if item['name'] == 'check_menu_items':
                                        result = check_menu_items(arguments['items'])
                                    elif item['name'] == 'process_order':
                                        result = process_order(arguments['order'], stream_sid)
                                    else:
                                        continue  # Skip unknown function calls

                                    function_call_output = {
                                        "type": "conversation.item.create",
                                        "item": {
                                            "type": "function_call_output",
                                            "call_id": item["call_id"],
                                            "output": result  # Needs to be a string
                                        }
                                    }
                                    await openai_ws.send(json.dumps(function_call_output))
                                    # define a response_create object to trigger the response
                                    response_create = {
                                        "type": "response.create"
                                    }
                                    await openai_ws.send(json.dumps(response_create))
                                except Exception as e:
                                    print(f"Error in function call {item['name']}: {e}")
                        await asyncio.sleep(0.5)
            except Exception as e:
                print(f"Error in send_to_twilio: {e}")

        await asyncio.gather(receive_from_twilio(), send_to_twilio())

async def process_order_response(openai_response):
    """Process the OpenAI response to extract and print the 'order' part."""
    try:
        # Extract the "order" from the response
        order_data = openai_response.get('order')
        if order_data:
            print("Order Details:")
            print(json.dumps(order_data, indent=4))
        else:
            print("No order found in the response.")
    except Exception as e:
        print(f"Error extracting order: {e}")

async def send_session_update(openai_ws):
    """Send session update to OpenAI WebSocket."""
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
            "tools": TOOLS,
            "tool_choice": "auto",
        }
    }
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
