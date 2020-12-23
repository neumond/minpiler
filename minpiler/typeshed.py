from typing import Optional, Union


class MConst:
    pass


class MObject:
    pass


Value = Union[float, MObject, MConst]
ColorComponent = float  # 0..255
Boolean = float  # 0 or 1
Angle = float  # 0..360
Integer = float  # number will be casted to int


def print(*args: Value) -> None:
    pass


def __print_flush(target: MObject) -> None:
    # this function is available only as print.flush
    pass


print.flush = __print_flush


def exit() -> None:
    pass


def min(a: float, b: float) -> float:
    pass


def max(a: float, b: float) -> float:
    pass


def atan2(x: float, y: float) -> Angle:
    pass


def dst(x: float, y: float) -> float:
    "Distance from (0, 0) to (x, y)"


def noise(x: float, y: float) -> float:
    "Simplex noise"


def abs(v: float) -> float:
    pass


def log(v: float) -> float:
    pass


def log10(v: float) -> float:
    pass


def sin(v: Angle) -> float:
    pass


def cos(v: Angle) -> float:
    pass


def tan(v: Angle) -> float:
    pass


def floor(v: float) -> Integer:
    pass


def ceil(v: float) -> Integer:
    pass


def sqrt(v: float) -> float:
    pass


def rand(range_: float) -> float:
    pass


def GetLink(index: int) -> Optional[MObject]:
    pass


def Radar(
        unit: MObject,
        target1: MConst, target2: MConst, target3: MConst,
        sort_type: MConst,
        sort_dir: MConst) -> Optional[MObject]:
    pass


def Sensor(unit: MObject, prop: MConst) -> Value:
    pass


class Draw:
    @staticmethod
    def clear(
            r: ColorComponent,
            g: ColorComponent,
            b: ColorComponent) -> None:
        pass

    @staticmethod
    def color(
            r: ColorComponent,
            g: ColorComponent,
            b: ColorComponent,
            a: ColorComponent) -> None:
        pass

    @staticmethod
    def stroke(width: float) -> None:
        pass

    @staticmethod
    def line(x: float, y: float, x2: float, y2: float) -> None:
        pass

    @staticmethod
    def rect(x: float, y: float, width: float, height: float) -> None:
        pass

    @staticmethod
    def lineRect(x: float, y: float, width: float, height: float) -> None:
        pass

    @staticmethod
    def poly(
            x: float, y: float,
            sides: float, radius: float, rotation: float) -> None:
        pass

    @staticmethod
    def linePoly(
            x: float, y: float,
            sides: float, radius: float, rotation: float) -> None:
        pass

    @staticmethod
    def triangle(
            x: float, y: float,
            x2: float, y2: float,
            x3: float, y3: float) -> None:
        pass

    @staticmethod
    def image(x: float, y: float, image: MConst, size: float, rotation: float):
        pass

    @staticmethod
    def flush(target: MObject) -> None:
        pass


class Control:
    @staticmethod
    def setEnabled(unit: MObject, is_enabled: Boolean) -> None:
        pass

    @staticmethod
    def shootPosition(unit: MObject, x: float, y: float) -> None:
        pass

    @staticmethod
    def shootObject(unit: MObject, target: MObject) -> None:
        pass

    @staticmethod
    def stopShooting(unit: MObject) -> None:
        pass

    @staticmethod
    def configure(unit: MObject, value: MConst) -> None:
        pass


# List of constants
# https://github.com/Anuken/Mindustry/blob/master/core/src/mindustry/logic/GlobalConstants.java#L15


# https://github.com/Anuken/Mindustry/blob/master/core/src/mindustry/content/Items.java
class Material:
    copper: MConst
    lead: MConst
    metaglass: MConst
    graphite: MConst
    sand: MConst
    coal: MConst
    titanium: MConst
    thorium: MConst
    scrap: MConst
    silicon: MConst
    plastanium: MConst
    phase_fab: MConst
    surge_alloy: MConst
    spore_pod: MConst
    blast_compound: MConst
    pyratite: MConst


# https://github.com/Anuken/Mindustry/blob/master/core/src/mindustry/content/Liquids.java
class Liquid:
    water: MConst
    slag: MConst
    oil: MConst
    cryofluid: MConst


# https://github.com/Anuken/Mindustry/blob/master/core/src/mindustry/logic/LAccess.java
class Property:
    totalItems: MConst
    firstItem: MConst
    totalLiquids: MConst
    totalPower: MConst
    itemCapacity: MConst
    liquidCapacity: MConst
    powerCapacity: MConst
    powerNetStored: MConst
    powerNetCapacity: MConst
    powerNetIn: MConst
    powerNetOut: MConst
    ammo: MConst
    ammoCapacity: MConst
    health: MConst
    maxHealth: MConst
    heat: MConst
    efficiency: MConst
    rotation: MConst
    x: MConst
    y: MConst
    shootX: MConst
    shootY: MConst
    shooting: MConst
    mineX: MConst
    mineY: MConst
    mining: MConst
    team: MConst
    type: MConst
    flag: MConst
    controlled: MConst
    commanded: MConst
    name: MConst
    config: MConst
    payloadCount: MConst
    payloadType: MConst

    enabled: MConst
    configure: MConst


# https://github.com/Anuken/Mindustry/blob/master/core/src/mindustry/logic/RadarSort.java
class Sort:
    asc: MConst
    desc: MConst

    distance: MConst
    health: MConst
    shield: MConst
    armor: MConst
    maxHealth: MConst


# https://github.com/Anuken/Mindustry/blob/master/core/src/mindustry/logic/RadarTarget.java
class Target:
    any: MConst
    enemy: MConst
    ally: MConst
    player: MConst
    attacker: MConst
    flying: MConst
    boss: MConst
    ground: MConst


# https://github.com/Anuken/Mindustry/blob/master/core/src/mindustry/content/UnitTypes.java
class UnitType:
    dagger: MConst
    mace: MConst
    fortress: MConst
    scepter: MConst
    reign: MConst
    nova: MConst
    pulsar: MConst
    quasar: MConst
    vela: MConst
    corvus: MConst
    crawler: MConst
    atrax: MConst
    spiroct: MConst
    arkyid: MConst
    toxopid: MConst
    flare: MConst
    horizon: MConst
    zenith: MConst
    antumbra: MConst
    eclipse: MConst
    mono: MConst
    poly: MConst
    mega: MConst
    quad: MConst
    oct: MConst
    risso: MConst
    minke: MConst
    bryde: MConst
    sei: MConst
    omura: MConst
    alpha: MConst
    beta: MConst
    gamma: MConst
    block: MConst


# https://github.com/Anuken/Mindustry/blob/master/core/src/mindustry/world/meta/BlockFlag.java
class BlockFlag:
    core: MConst
    storage: MConst
    generator: MConst
    turret: MConst
    factory: MConst
    repair: MConst
    rally: MConst
    battery: MConst
    resupply: MConst
    reactor: MConst
    unitModifier: MConst
    extinguisher: MConst


# https://github.com/Anuken/Mindustry/blob/master/core/src/mindustry/content/Blocks.java
class Block:
    # TODO: put here full list of blocks

    air: MConst
    solid: MConst
