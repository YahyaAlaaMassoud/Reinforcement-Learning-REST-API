from rc_car import move_forward
import sys

def move(duration):
    move_forward(duration)
    
if __name__ == "__main__":
    duration = int(sys.argv[1])
    move(duration)

