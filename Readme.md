# KoiWave - AI-Powered Restaurant Phone Ordering Solution

Transform your restaurant's phone ordering experience with KoiWave.

Our solution addresses common restaurant pain points while boosting revenue and customer satisfaction.
  - No staffing issues for phone orders
  - Eliminated order errors
  - Consistent reliable customer experience


## Demo

Check out the [demo video](https://www.youtube.com/watch?v=LNRGU5REvIU) to see KoiWave in action.


## How It Works

1. Customer calls the restaurant.
2. KoiWave AI assistant handles the order, just like a human front desk.
3. Customer receives an SMS with order confirmation and a payment link.
4. Restaurant receives the order details upon payment.


## Technical Prerequisites

- **Python 3.9+** 
  - [Download Python here](https://www.python.org/downloads/)
- **Twilio Account**
  - [Sign up for Twilio](https://www.twilio.com/try-twilio)
  - Purchase a Twilio number with _Voice_ and _SMS_ capabilities ([Here are instructions](https://help.twilio.com/articles/223135247-How-to-Search-for-and-Buy-a-Twilio-Phone-Number-from-Console))
- **OpenAI Account**
  - [Sign up for OpenAI](https://platform.openai.com/)


## Setup Instructions

### 1. Configure Environment Variables

Copy the `.env.example` file to `.env`.

```bash
cp .env.example .env
```

In the `.env` file, update the fields with your actual values.
```
OPENAI_API_KEY=your_api_key
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=your_twilio_phone_number
PORT=your_port
```

### 2. Open an ngrok tunnel

Forward requests to your local server using `ngrok`. ([Download `ngrok` here](https://ngrok.com/)).

```bash
ngrok http --domain=[your-ngrok-subdomain].ngrok.app 5050
```

Note that the ngrok command above forwards to a development server running on port 5050, which is the default port configured in this application. If you override the `PORT` defined above, you will need to update the ngrok command accordingly.


### 3. Create virtual environment

Create the Environment with required dependencies.

```bash
conda env create -f environment.yml
```

Activate the created environment.

```bash
conda activate koiwave
```

### 4. Twilio Setup

#### Point a Phone Number to your ngrok URL
In the [Twilio Console](https://console.twilio.com/), go to **Phone Numbers** > **Manage** > **Active Numbers** and click on the phone number you purchased for this app in the **Technical Prerequisites**.

In your Phone Number configuration settings, update the first **A call comes in** dropdown to **Webhook**, and paste your ngrok forwarding URL (referenced above), followed by `/incoming-call`. For example, `https://[your-ngrok-subdomain].ngrok.app/incoming-call`. Then, click **Save configuration**.

#### Enable SMS Service

Twilio may require a review process for SMS service activation. Please confirm approval with Twilio support.

### 5. Start the server

Once `.env` is set up, `ngrok` is running, and Twilio is configured properly, run the server in the virtual environment:

```bash
python main.py
```

## Menu Configuration

Menu (`menus/dragon_chinese_restaurant_menu.json`) is defined in JSON with information like:
- Restaurant information
- Tax rates
- Menu sections (e.g., Appetizers, Main Dishes)
- Item details (name, price, options)

Example menu structure:
```json
{
    "restaurant_name": "Restaurant Name",
    "tax_rate": 0.0775,
    "menu": [
        {
            "section": "Section Name",
            "items": [
                {
                    "id": 1,
                    "name": "Item Name",
                    "price": 10.99
                }
            ]
        }
    ]
}
```

## Testing

Once the server is running, call your Twilio number to interact with the AI Assistant and place an order. You’ll receive an SMS confirmation with a payment link. Enjoy!

## License

MIT License - see LICENSE file for details

## Acknowledgements

Thanks to [speech-assistant-openai-realtime-api-python](https://github.com/twilio-samples/speech-assistant-openai-realtime-api-python) for guidance on Twilio and OpenAI API integration.
