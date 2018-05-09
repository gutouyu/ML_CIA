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



log_eta_001_x, log_eta_001_y = parse_log('ffm.eta.0.01')
log_eta_002_x, log_eta_002_y = parse_log('ffm.eta.0.02')
log_eta_005_x, log_eta_005_y = parse_log('ffm.eta.0.05')
log_eta_01_x, log_eta_01_y = parse_log('ffm.eta.0.1')
log_eta_02_x, log_eta_02_y = parse_log('ffm.eta.0.2')
log_eta_05_x, log_eta_05_y = parse_log('ffm.eta.0.5')


plt.plot(log_eta_001_x, log_eta_001_y)
plt.plot(log_eta_002_x, log_eta_002_y)
plt.plot(log_eta_005_x, log_eta_005_y)
plt.plot(log_eta_01_x, log_eta_01_y)
plt.plot(log_eta_02_x, log_eta_02_y)
plt.plot(log_eta_05_x, log_eta_05_y)

plt.ylim([0.454, 0.5])
plt.xlim([1, 25])
plt.legend([r'$\eta = 0.01$', r'$\eta = 0.02$', r'$\eta = 0.05$', r'$\eta = 0.1$', r'$\eta = 0.2$', r'$\eta = 0.5$'])
plt.xlabel('Epochs')
plt.ylabel('Logloss')
plt.grid()
plt.savefig('criteo_impact_eta.png')
