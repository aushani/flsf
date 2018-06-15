from evaluate import *

if len(sys.argv) > 2:
    print 'Need model file!'

model_file = sys.argv[1]
match, score = evaluate(model_file, fl=True)

np.savetxt('fl_match.csv', match)
np.savetxt('fl_score.csv', score)
