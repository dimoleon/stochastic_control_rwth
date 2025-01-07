T = 10 
M = Int(1e5) 
N = Int(10)

MINY = 1
MINZ = 1
MAXY = 5 
MAXZ = 10 

RGOAL = 50
ROBST = -10 
RDIAG = -6 

states = [(y, z) for z ∈ MINZ:MAXZ for y ∈ MINY:MAXY]
goals = Set([(2, 10), (4, 10)])
obstacles = Set([(2, 2), (4, 2), 
                (3, 3), 
                (1, 4), (4, 4), 
                (3, 5), (5, 5), 
                (1, 6), (4, 6),
                (2, 7), (5, 7), 
                (1, 8), (3, 8), (4, 8), 
                (3, 9), (4, 9),
                (1, 10), (3, 10), (5, 10)]) 

actions = [1, 2, 3]
# 1 -> same
# 2 -> up 
# 3 -> down 
# 0 -> no action possible 

valid_init_states = []
for x ∈ states 
    if x ∈ obstacles || x ∈ goals
        continue
    end
    push!(valid_init_states, x) 
end
