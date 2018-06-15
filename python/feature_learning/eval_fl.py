from evaluate import *

if len(sys.argv) > 2:
    print 'Need model file!'

model_file = sys.argv[1]
match, score = evaluate(model_file, ml=True)

np.savetxt('ml_match.csv', match)
np.savetxt('ml_score.csv', score)
