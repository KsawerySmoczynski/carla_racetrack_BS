import carla

from control.abstract_control import Controller

#NAGRODA (Dystans do - Dystans przejechany)

class Agent:
    def __init__(self, world:carla.World, controller:Controller, vehicle:str, sensors:dict):
        '''

        :param world:
        :param controller:
        :param vehicle:
        :param sensors:
        '''
        self.world = world
        self.actor:carla.Vehicle = self.world.get_blueprint_library().find(vehicle)
        self.controller = controller
        self.sensors = sensors

    def play_step(self):
        pass

    def initialize_environment(self):
        pass


    #TODO add dunder methods __str__


