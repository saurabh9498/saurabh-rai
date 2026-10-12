"""CLI client for the Conversational AI API."""

import argparse
import asyncio
import json
import websockets
import httpx
import sys


async def text_mode(api_url: str, session_id: str):
    """Interactive text mode."""
    print("Text mode. Type 'quit' to exit.")
    
    async with httpx.AsyncClient() as client:
        while True:
            try:
                text = input("\nYou: ").strip()
                if text.lower() == 'quit':
                    break
                
                response = await client.post(
                    f"{api_url}/chat",
                    json={"text": text, "session_id": session_id},
                )
                
                data = response.json()
                print(f"Bot: {data['text']}")
                print(f"  [Intent: {data['intent']} ({data['confidence']:.2f})]")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


async def voice_mode(ws_url: str, session_id: str):
    """Voice mode with WebSocket streaming."""
    print("Voice mode. Press Ctrl+C to exit.")
    
    try:
        async with websockets.connect(f"{ws_url}/ws/voice/{session_id}") as ws:
            print("Connected to voice server")
            
            # In a real implementation, this would capture audio from microphone
            # and stream it to the server
            
            while True:
                message = await ws.recv()
                data = json.loads(message)
                
                if data['type'] == 'transcription':
                    print(f"[Transcription]: {data['text']}")
                elif data['type'] == 'response':
                    print(f"Bot: {data['text']}")
                    
    except KeyboardInterrupt:
        print("\nDisconnected")


def main():
    parser = argparse.ArgumentParser(description="Conversational AI Client")
    parser.add_argument("--mode", choices=["text", "voice"], default="text")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--session", default="cli-session")
    
    args = parser.parse_args()
    
    if args.mode == "text":
        asyncio.run(text_mode(args.url, args.session))
    else:
        ws_url = args.url.replace("http", "ws")
        asyncio.run(voice_mode(ws_url, args.session))


if __name__ == "__main__":
    main()
