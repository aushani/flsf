from evaluate import *

if len(sys.argv) > 2:
    print 'Need model file!'

model_file = sys.argv[1]
match, score = evaluate(model_file, il=True)

np.savetxt('il_match.csv', match)
np.savetxt('il_score.csv', score)
