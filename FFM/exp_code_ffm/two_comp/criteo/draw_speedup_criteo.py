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



log_thread_1_x, log_thread_1_y = parse_log('ffm.thread.1')
log_thread_2_x, log_thread_2_y = parse_log('ffm.thread.2')
log_thread_4_x, log_thread_4_y = parse_log('ffm.thread.4')
log_thread_6_x, log_thread_6_y = parse_log('ffm.thread.6')
log_thread_8_x, log_thread_8_y = parse_log('ffm.thread.8')
log_thread_10_x, log_thread_10_y = parse_log('ffm.thread.10')
log_thread_12_x, log_thread_12_y = parse_log('ffm.thread.12')


x = [1, 2, 4, 6, 8, 10, 12]
base = max(log_thread_1_y)
speedup_thread_1 = base
speedup_thread_2 = base*1.0/max(log_thread_2_y)
speedup_thread_4 = base*1.0/max(log_thread_4_y)
speedup_thread_6 = base*1.0/max(log_thread_6_y)
speedup_thread_8 = base*1.0/max(log_thread_8_y)
speedup_thread_10 = base*1.0/max(log_thread_10_y)
speedup_thread_12 = base*1.0/max(log_thread_12_y)


y = [speedup_thread_1, speedup_thread_2, speedup_thread_4, speedup_thread_6, speedup_thread_8, speedup_thread_10, speedup_thread_12]

plt.plot(x, y)

plt.xlim([1, 12])
plt.xlabel('#threads')
plt.ylabel('Speedup')
plt.grid()
plt.xticks(x, ['1', '2', '4', '6', '8', '10', '12'])
plt.savefig('criteo_speedup.png')
