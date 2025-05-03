import pygame 
from .interface import GameInterface
from .settings import GameSettings 

def main():
    pygame.init()
    pygame.mixer.init()
    settings = GameSettings()
    interface = GameInterface(settings)
    interface.run()

if __name__ == "__main__":
    main()