# [Part 7](https://youtu.be/_gTYJ1KTOxw)
```python
async def intel(self):
    # for game_info: https://github.com/Dentosal/python-sc2/blob/master/sc2/game_info.py#L162
    print(self.game_info.map_size)
    # flip around. It's y, x when you're dealing with an array.
    game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)
    for nexus in self.units(NEXUS):
        nex_pos = nexus.position
        print(nex_pos)
        cv2.circle(game_data, (int(nex_pos[0]), int(nex_pos[1])), 10, (0, 255, 0), -1)  # BGR

    # flip horizontally to make our final fix in visual representation:
    flipped = cv2.flip(game_data, 0)
    resized = cv2.resize(flipped, dsize=None, fx=2, fy=2)

    cv2.imshow('Intel', resized)
    cv2.waitKey(1)
```
# [Part 8](https://www.youtube.com/watch?v=HOwwgu_xDKk)
```python
async def scout(self):
    if len(self.units(OBSERVER)) > 0:
        scout = self.units(OBSERVER)[0]
        if scout.is_idle:
            enemy_location = self.enemy_start_locations[0]
            move_to = self.random_location_variance(enemy_location)
            print(move_to)
            await self.do(scout.move(move_to))

    else:
        for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
            if self.can_afford(OBSERVER) and self.supply_left > 0:
                await self.do(rf.train(OBSERVER))
```
# [Part 9](https://www.youtube.com/watch?v=ycmgbUd8LQI)
```python
async def intel(self):
    game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

    # UNIT: [SIZE, (BGR COLOR)]
    '''from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
CYBERNETICSCORE, STARGATE, VOIDRAY'''
    draw_dict = {
        NEXUS: [15, (0, 255, 0)],
        PYLON: [3, (20, 235, 0)],
        PROBE: [1, (55, 200, 0)],
        ASSIMILATOR: [2, (55, 200, 0)],
        GATEWAY: [3, (200, 100, 0)],
        CYBERNETICSCORE: [3, (150, 150, 0)],
        STARGATE: [5, (255, 0, 0)],
        ROBOTICSFACILITY: [5, (215, 155, 0)],

        VOIDRAY: [3, (255, 100, 0)],
        #OBSERVER: [3, (255, 255, 255)],
    }

    for unit_type in draw_dict:
        for unit in self.units(unit_type).ready:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), draw_dict[unit_type][0], draw_dict[unit_type][1], -1)

    main_base_names = ["nexus", "supplydepot", "hatchery"]
    for enemy_building in self.known_enemy_structures:
        pos = enemy_building.position
        if enemy_building.name.lower() not in main_base_names:
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), 5, (200, 50, 212), -1)
    for enemy_building in self.known_enemy_structures:
        pos = enemy_building.position
        if enemy_building.name.lower() in main_base_names:
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), 15, (0, 0, 255), -1)

    for enemy_unit in self.known_enemy_units:
        if not enemy_unit.is_structure:
            worker_names = ["probe", "scv", "drone"]
            # if that unit is a PROBE, SCV, or DRONE... it's a worker
            pos = enemy_unit.position
            if enemy_unit.name.lower() in worker_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (55, 0, 155), -1)
            else:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (50, 0, 215), -1)

    for obs in self.units(OBSERVER).ready:
        pos = obs.position
        cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (255, 255, 255), -1)

    line_max = 50
    mineral_ratio = self.minerals / 1500
    if mineral_ratio > 1.0:
        mineral_ratio = 1.0
    vespene_ratio = self.vespene / 1500
    if vespene_ratio > 1.0:
        vespene_ratio = 1.0
    population_ratio = self.supply_left / self.supply_cap
    if population_ratio > 1.0:
        population_ratio = 1.0
    plausible_supply = self.supply_cap / 200.0
    military_weight = len(self.units(VOIDRAY)) / (self.supply_cap-self.supply_left)
    if military_weight > 1.0:
        military_weight = 1.0

    cv2.line(game_data, (0, 19), (int(line_max*military_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
    cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
    cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
    cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
    cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500

    # flip horizontally to make our final fix in visual representation:
    flipped = cv2.flip(game_data, 0)
    resized = cv2.resize(flipped, dsize=None, fx=2, fy=2)
    cv2.imshow('Intel', resized)
    cv2.waitKey(1)
```
# [Part 10](https://www.youtube.com/watch?v=lCTn3dVc1_M)
- sequential model
- add in all the layers to the model
- Conv2D
  - 3rd dimension is channels, color is not 3rd dimension
  - padding, shifting window to get data, what do you do with edge
  - dropout
    - avoid bias nodes in the network
  - learning rate
  - pythonprogramming.net convolutional neural network to learn more
# [Part 11](https://www.youtube.com/watch?v=IgnYjgpGSzE)
- load data into vram for tensor flow
- class balancing important to balance bias of choices
  - quick descent/ascent if not balance
# [Part 12](https://www.youtube.com/watch?v=D5c2xJQH3Ag)
- part 11 = built neural network
- part 12 = use it
- change random choice to attack to use the model
- about to clean up code
# [Part 13](https://www.youtube.com/watch?v=zt97GlmjQbY)
- tracking game time exactly
- use unit radius
- use grayscale instead of color for intel image
# [Part 14](https://www.youtube.com/watch?v=81JULbBnv0A)
- improving scouting
- putting obs on every expansion
- scouting with 1 probe at a time until getting obs
# [Part 15](https://www.youtube.com/watch?v=eOv1aPRE1jo&index=15&list=PLQVvvaa0QuDcT3tPehHdisGMc8TInNqdq)
- random 14 choices
# [Part 16](https://www.youtube.com/watch?v=6N1bsDNAIB8&list=PLQVvvaa0QuDcT3tPehHdisGMc8TInNqdq&index=16)
- vizualization changes
- gray scale, opponent darker
- radius not fill
- asking to figure out alpha
# [Part 17](https://www.youtube.com/watch?v=oi6QBUZUgbc&index=17&list=PLQVvvaa0QuDcT3tPehHdisGMc8TInNqdq)
- results from training the model
- he is randomly choosing when to build workers
  - instead, we should randomly decide at the beginning at what worker count to stop at for some amount of time
  - all workers or all units (works well for zerg)
- looking at tensor boards
- not beating random with model