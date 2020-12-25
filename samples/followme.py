from void import (
    M,
    duo1,
    leader,
)


if leader is None:
    leader = duo1.radar(
        M.at.player, M.at.any, M.at.any,
        M.at.distance, M.at.asc)
elif leader.health <= 0:
    leader = None
else:
    M.unit.bind(M.at.nova)
    M.unit.approach(leader.x, leader.y, 5)
    M.unit.targetPosition(leader.shootX, leader.shootY, leader.shooting)
