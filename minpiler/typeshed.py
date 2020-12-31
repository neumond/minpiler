from typing import Optional, Tuple, Union


class MConst: ...  # noqa: E701


Value = Union[float, 'MObject', MConst]
ColorComponent = float  # 0..255
Boolean = float  # 0 or 1
Angle = float  # 0..360
Integer = float  # number will be casted to int


class MObject:
    def drawFlush(self) -> None: ...
    def printFlush(self) -> None: ...

    def radar(self, target1: MConst, target2: MConst, target3: MConst, sort_type: MConst, sort_dir: MConst) -> Optional['MObject']: ...  # noqa: E501
    def sensor(self, prop: MConst) -> Value: ...

    def setEnabled(self, is_enabled: Boolean) -> None: ...
    def targetPosition(self, x: float, y: float, shoot: Boolean) -> None: ...
    def targetObject(self, target: 'MObject', shoot: Boolean) -> None: ...
    def configure(self, value: MConst) -> None: ...


class M:
    @staticmethod
    def print(*args: Value) -> None: ...
    @staticmethod
    def exit() -> None: ...
    @staticmethod
    def min(a: float, b: float) -> float: ...
    @staticmethod
    def max(a: float, b: float) -> float: ...
    @staticmethod
    def atan2(x: float, y: float) -> Angle: ...
    @staticmethod
    def dst(x: float, y: float) -> float: ...
    @staticmethod
    def noise(x: float, y: float) -> float: ...
    @staticmethod
    def abs(v: float) -> float: ...
    @staticmethod
    def log(v: float) -> float: ...
    @staticmethod
    def log10(v: float) -> float: ...
    @staticmethod
    def sin(v: Angle) -> float: ...
    @staticmethod
    def cos(v: Angle) -> float: ...
    @staticmethod
    def tan(v: Angle) -> float: ...
    @staticmethod
    def floor(v: float) -> Integer: ...
    @staticmethod
    def ceil(v: float) -> Integer: ...
    @staticmethod
    def sqrt(v: float) -> float: ...
    @staticmethod
    def rand(range_: float) -> float: ...
    @staticmethod
    def linkCount() -> Integer: ...
    @staticmethod
    def getLink(index: Integer) -> Optional[MObject]: ...

    class draw:
        @staticmethod
        def clear(r: ColorComponent, g: ColorComponent, b: ColorComponent) -> None: ...  # noqa: E501
        @staticmethod
        def color(r: ColorComponent, g: ColorComponent, b: ColorComponent, a: ColorComponent) -> None: ...  # noqa: E501
        @staticmethod
        def stroke(width: float) -> None: ...
        @staticmethod
        def line(x: float, y: float, x2: float, y2: float) -> None: ...
        @staticmethod
        def rect(x: float, y: float, width: float, height: float) -> None: ...
        @staticmethod
        def lineRect(x: float, y: float, width: float, height: float) -> None: ...  # noqa: E501
        @staticmethod
        def poly(x: float, y: float, sides: float, radius: float, rotation: float) -> None: ...  # noqa: E501
        @staticmethod
        def linePoly(x: float, y: float, sides: float, radius: float, rotation: float) -> None: ...  # noqa: E501
        @staticmethod
        def triangle(x: float, y: float, x2: float, y2: float, x3: float, y3: float) -> None: ...  # noqa: E501
        @staticmethod
        def image(x: float, y: float, image: MConst, size: float, rotation: float): ...  # noqa: E501

    class unit:
        @staticmethod
        def bind(utype: MConst) -> None: ...
        @staticmethod
        def radar(target1: MConst, target2: MConst, target3: MConst, sort_type: MConst, sort_dir: MConst) -> Optional[MObject]: ...  # noqa: E501
        @staticmethod
        def stop() -> None: ...
        @staticmethod
        def move(x: float, y: float) -> None: ...
        @staticmethod
        def approach(x: float, y: float, radius: float) -> None: ...
        @staticmethod
        def boost(value: float) -> None: ...
        @staticmethod
        def pathfind() -> None: ...
        @staticmethod
        def targetPosition(x: float, y: float, shoot: Boolean) -> None: ...
        @staticmethod
        def targetObject(unit: MObject, shoot: Boolean) -> None: ...
        @staticmethod
        def itemDrop(target: MObject, amount: float) -> None: ...
        @staticmethod
        def itemTake(target: MObject, material: MConst, amount: float) -> None: ...  # noqa: E501
        @staticmethod
        def payDrop() -> None: ...
        @staticmethod
        def payTake(amount: float) -> None: ...
        @staticmethod
        def mine(x: float, y: float) -> None: ...
        @staticmethod
        def setFlag(value: float) -> None: ...
        @staticmethod
        def build(x: float, y: float, block: MConst, rotation: Angle, config: MConst) -> None: ...  # noqa: E501
        @staticmethod
        def getBlock(x: float, y: float) -> Tuple[MConst, MObject]: ...
        @staticmethod
        def within(x: float, y: float, radius: float) -> Boolean: ...

    class locate:
        @staticmethod
        def building(block_type: MConst, enemy: Boolean) -> Tuple[Boolean, float, float, MObject]: ...  # noqa: E501
        @staticmethod
        def ore(material: MConst) -> Tuple[Boolean, float, float]: ...
        @staticmethod
        def spawn() -> Tuple[Boolean, float, float, MObject]: ...
        @staticmethod
        def damaged() -> Tuple[Boolean, float, float, MObject]: ...

    class at:
        unit: MObject = ...

        # List of constants
        # https://github.com/Anuken/Mindustry/blob/master/core/src/mindustry/logic/GlobalConstants.java#L15

        # https://github.com/Anuken/Mindustry/blob/master/core/src/mindustry/content/Items.java
        copper: MConst = ...
        lead: MConst = ...
        metaglass: MConst = ...
        graphite: MConst = ...
        sand: MConst = ...
        coal: MConst = ...
        titanium: MConst = ...
        thorium: MConst = ...
        scrap: MConst = ...
        silicon: MConst = ...
        plastanium: MConst = ...
        phase_fab: MConst = ...
        surge_alloy: MConst = ...
        spore_pod: MConst = ...
        blast_compound: MConst = ...
        pyratite: MConst = ...

        # https://github.com/Anuken/Mindustry/blob/master/core/src/mindustry/content/Liquids.java
        water: MConst = ...
        slag: MConst = ...
        oil: MConst = ...
        cryofluid: MConst = ...

        # https://github.com/Anuken/Mindustry/blob/master/core/src/mindustry/logic/LAccess.java
        totalItems: MConst = ...
        firstItem: MConst = ...
        totalLiquids: MConst = ...
        totalPower: MConst = ...
        itemCapacity: MConst = ...
        liquidCapacity: MConst = ...
        powerCapacity: MConst = ...
        powerNetStored: MConst = ...
        powerNetCapacity: MConst = ...
        powerNetIn: MConst = ...
        powerNetOut: MConst = ...
        ammo: MConst = ...
        ammoCapacity: MConst = ...
        health: MConst = ...
        maxHealth: MConst = ...
        heat: MConst = ...
        efficiency: MConst = ...
        rotation: MConst = ...
        x: MConst = ...
        y: MConst = ...
        shootX: MConst = ...
        shootY: MConst = ...
        shooting: MConst = ...
        mineX: MConst = ...
        mineY: MConst = ...
        mining: MConst = ...
        team: MConst = ...
        type: MConst = ...
        flag: MConst = ...
        controlled: MConst = ...
        commanded: MConst = ...
        name: MConst = ...
        config: MConst = ...
        payloadCount: MConst = ...
        payloadType: MConst = ...

        enabled: MConst = ...
        configure: MConst = ...

        # https://github.com/Anuken/Mindustry/blob/master/core/src/mindustry/logic/RadarSort.java
        asc: MConst = ...
        desc: MConst = ...

        distance: MConst = ...
        health: MConst = ...
        shield: MConst = ...
        armor: MConst = ...
        maxHealth: MConst = ...

        # https://github.com/Anuken/Mindustry/blob/master/core/src/mindustry/logic/RadarTarget.java
        any: MConst = ...
        enemy: MConst = ...
        ally: MConst = ...
        player: MConst = ...
        attacker: MConst = ...
        flying: MConst = ...
        boss: MConst = ...
        ground: MConst = ...

        # https://github.com/Anuken/Mindustry/blob/master/core/src/mindustry/content/UnitTypes.java
        dagger: MConst = ...
        mace: MConst = ...
        fortress: MConst = ...
        scepter: MConst = ...
        reign: MConst = ...
        nova: MConst = ...
        pulsar: MConst = ...
        quasar: MConst = ...
        vela: MConst = ...
        corvus: MConst = ...
        crawler: MConst = ...
        atrax: MConst = ...
        spiroct: MConst = ...
        arkyid: MConst = ...
        toxopid: MConst = ...
        flare: MConst = ...
        horizon: MConst = ...
        zenith: MConst = ...
        antumbra: MConst = ...
        eclipse: MConst = ...
        mono: MConst = ...
        poly: MConst = ...
        mega: MConst = ...
        quad: MConst = ...
        oct: MConst = ...
        risso: MConst = ...
        minke: MConst = ...
        bryde: MConst = ...
        sei: MConst = ...
        omura: MConst = ...
        alpha: MConst = ...
        beta: MConst = ...
        gamma: MConst = ...
        block: MConst = ...

        # https://github.com/Anuken/Mindustry/blob/master/core/src/mindustry/world/meta/BlockFlag.java
        core: MConst = ...
        storage: MConst = ...
        generator: MConst = ...
        turret: MConst = ...
        factory: MConst = ...
        repair: MConst = ...
        rally: MConst = ...
        battery: MConst = ...
        resupply: MConst = ...
        reactor: MConst = ...
        unitModifier: MConst = ...
        extinguisher: MConst = ...

        # https://github.com/Anuken/Mindustry/blob/master/core/src/mindustry/content/Blocks.java
        # TODO: put here full list of blocks
        air: MConst = ...
        solid: MConst = ...
