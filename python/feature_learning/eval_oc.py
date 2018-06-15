from evaluate import *

if len(sys.argv) > 2:
    print 'Need model file!'

model_file = sys.argv[1]
match, score = evaluate(model_file, oc=True)

np.savetxt('oc_match.csv', match)
np.savetxt('oc_score.csv', score)
