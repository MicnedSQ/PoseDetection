import websocket
import threading

def on_open(ws):
    threading.Thread(target=get_user_input, args=(ws,)).start()

def get_user_input(ws):
    while True:
        user_input = input("Enter direction (UP/DOWN/LEFT/RIGHT): ")
        if user_input.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]:
            ws.send(user_input.upper())
        else:
            print("Invalid input. Please enter UP, DOWN, LEFT, or RIGHT.")

if __name__ == "__main__":
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("ws://localhost:8080/Auth")
    ws.on_open = on_open
    ws.run_forever()
