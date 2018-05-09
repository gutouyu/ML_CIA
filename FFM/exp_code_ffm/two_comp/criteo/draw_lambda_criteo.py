import matplotlib.pyplot as plt

def parse_log(log_path):
    x = []
    y = []

    for line_idx, line in enumerate(open(log_path)):
        if line_idx == 0: continue
        epoch, tr_logloss, va_logloss, time = map(float, line.split())
        x.append(int(epoch))
        y.append(va_logloss)

    return x, y



log_lambda_1e6_x, log_lambda_1e6_y = parse_log('ffm.lambda.0.000001')
log_lambda_1e5_x, log_lambda_1e5_y = parse_log('ffm.lambda.0.00001')
log_lambda_1e4_x, log_lambda_1e4_y = parse_log('ffm.lambda.0.0001')
log_lambda_1e3_x, log_lambda_1e3_y = parse_log('ffm.lambda.0.001')


plt_l_1e6 = plt.plot(log_lambda_1e6_x, log_lambda_1e6_y)
plt_l_1e5 = plt.plot(log_lambda_1e5_x, log_lambda_1e5_y)
plt_l_1e4 = plt.plot(log_lambda_1e4_x, log_lambda_1e4_y)
plt_l_1e3 = plt.plot(log_lambda_1e3_x, log_lambda_1e3_y)

plt.ylim([0.453, 0.6])
plt.xlim([1, 140])
plt.legend([r'$\lambda = 1e-6$', r'$\lambda = 1e-5$', r'$\lambda = 1e-4$', r'$\lambda = 1e-3$'])
plt.xlabel('Epochs')
plt.ylabel('Logloss')
plt.grid()
plt.savefig('criteo_impact_lambda.png')
