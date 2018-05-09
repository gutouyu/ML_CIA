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


plt.plot(log_thread_1_x, log_thread_1_y)
plt.plot(log_thread_2_x, log_thread_2_y)
plt.plot(log_thread_4_x, log_thread_4_y)
plt.plot(log_thread_6_x, log_thread_6_y)
plt.plot(log_thread_8_x, log_thread_8_y)
plt.plot(log_thread_10_x, log_thread_10_y)
plt.plot(log_thread_12_x, log_thread_12_y)

plt.ylim([0.456, 0.47])
plt.xlim([1, 7])
plt.legend(['1 thread', '2 threads', '4 threads', '6 threads', '8 threads', '10 threads', '12 threads'])
plt.xlabel('Epochs')
plt.ylabel('Logloss')
plt.grid()
plt.savefig('criteo_impact_thread.png')
