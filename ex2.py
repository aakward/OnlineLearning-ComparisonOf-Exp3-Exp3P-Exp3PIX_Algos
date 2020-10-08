import math
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

np.random.seed(1000)
T=input("Enter the value of T: ")     #time horizon
def main1():
    
    K=10            #no. of arms
    eta=0           #learning rate
    c=0.1
    loss1_param=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.4,0.6]
    loss2_param=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.4,0.3]
    arms=[i for i in range(10)]
    R_eta=[]
    exp3_regret=[]
    while(c<=2.1):
        eta=c*math.sqrt(2.0*math.log(K)/(K*T))
        c=c+0.2
        R=[]
        b=[]
        b.append(eta)        
        for i in range(50):
            P=[]
            P.append([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
            regret=0
            cum_loss=[0 for i in range(K)]
            reg_loss=0
            for t in range(T):
                a=[]
                I_t=np.random.choice(arms,1,p=P[t])
                if(t<T/2):
                    l=np.random.binomial(1,loss1_param[I_t])
                    reg_loss=reg_loss+loss1_param[I_t]
                else:
                    l=np.random.binomial(1,loss2_param[I_t])
                    reg_loss=reg_loss+loss2_param[I_t]

                cum_loss[I_t]=cum_loss[I_t]+l/P[t][I_t]
                s=0                
                for j in range(K):
                    temp=0
                    temp=math.exp(-1.0*eta*cum_loss[j])
                    a.append(temp)
                    s=s+temp
                for j in range(K):
                    a[j]=a[j]/s
                P.append(a)

            reg_loss=reg_loss-(T*0.4)
            b.append(reg_loss)
            exp3_regret.append(b)
        
        R_eta.append(sum(R)/50)
    print R_eta
    eta = []
    regret_mean = []
    regret_err = []
    freedom_degree = len(exp3_regret[0]) - 2
    for regret in exp3_regret:
        eta.append(regret[0])
        regret_mean.append(np.mean(regret[1:]))
        regret_err.append(ss.t.ppf(0.95, freedom_degree) * ss.sem(regret[1:]))

    colors = list("rgbcmyk")
    shape = ['--^', '--d', '--v']
    plt.errorbar(eta, regret_mean, regret_err, color=colors[0])
    plt.plot(eta, regret_mean, colors[0] + shape[0], label='EXP3')
    print "EXP3 done!"
    
    


def main2():
   
    K=10            #no. of arms
    eta=0           #learning rate
    beta=0
    gamma=0    
    c=0.1
    loss1_param=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.4,0.6]
    loss2_param=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.4,0.3]
    arms=[i for i in range(10)]
    R_eta=[]
    exp3p_regret=[]
    while(c<=2.1):
        eta=c*math.sqrt(2.0*math.log(K)/(K*T))        
        c=c+0.2
        R=[]
        b=[]
        b.append(eta)
        beta=eta
        gamma=K*eta
        for i in range(50):
            P=[]
            P.append([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
            regret=0
            cum_loss=[0 for i in range(K)]
            reg_loss=0
            for t in range(T):
                a=[]
                I_t=np.random.choice(arms,1,p=P[t])
                if(t<T/2):
                    l=np.random.binomial(1,loss1_param[I_t])
                    reg_loss=reg_loss+loss1_param[I_t]
                else:
                    l=np.random.binomial(1,loss2_param[I_t])
                    reg_loss=reg_loss+loss2_param[I_t]
                for x in range(K):
                    if(x==I_t):                    
                        cum_loss[x]=cum_loss[x]+((l+beta)/P[t][x])
                    else:
                        cum_loss[x]=cum_loss[x]+(beta/P[t][x])
                s=0                                      
                for j in range(K):
                    temp=0
                    temp=math.exp(-1.0*eta*cum_loss[j])
                    a.append(temp)
                    s=s+temp
                for j in range(K):
                    a[j]=(1-gamma)*(a[j]/s)+(gamma/K)
                P.append(a)

            reg_loss=reg_loss-(T*0.4)
            b.append(reg_loss)
            R.append(reg_loss)
            exp3p_regret.append(b)
        
        R_eta.append(sum(R)/50)
    print R_eta
    eta = []
    regret_mean = []
    regret_err = []
    freedom_degree = len(exp3p_regret[0]) - 2
    for regret in exp3p_regret:
        eta.append(regret[0])
        regret_mean.append(np.mean(regret[1:]))
        regret_err.append(ss.t.ppf(0.95, freedom_degree) * ss.sem(regret[1:]))
    colors = list("rgbcmyk")
    shape = ['--^', '--d', '--v']
    plt.errorbar(eta, regret_mean, regret_err, color=colors[1])
    plt.plot(eta, regret_mean, colors[1] + shape[1], label='EXP3.P')
    print "EXP3P Done!"
    



def main3():
    K=10            #no. of arms
    eta=0           #learning rate
    gamma=0    
    c=0.1
    loss1_param=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.4,0.6]
    loss2_param=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.4,0.3]
    arms=[i for i in range(10)]
    R_eta=[]
    exp3ix_regret=[]
    while(c<=2.1):
        eta=c*math.sqrt(2.0*math.log(K)/(K*T))        
        c=c+0.2
        R=[]
        b=[]
        b.append(eta)
        gamma=eta/2
        for i in range(50):
            P=[]
        
            w=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
            regret=0
            cum_loss=[0 for i in range(K)]
            reg_loss=0
            for t in range(T):
                a=[]
                for x in range(K):
                    a.append(w[x]/sum(w))
                P.append(a)
                I_t=np.random.choice(arms,1,p=P[t])
                if(t<T/2):
                    l=np.random.binomial(1,loss1_param[I_t])
                    reg_loss=reg_loss+loss1_param[I_t]
                else:
                    l=np.random.binomial(1,loss2_param[I_t])
                    reg_loss=reg_loss+loss2_param[I_t]
                for x in range(K):
                    if (x==I_t):
                        w[x]=w[x]*math.exp(-1*eta*l/(P[t][I_t]+gamma))
            reg_loss=reg_loss-(T*0.4)
            b.append(reg_loss)
            R.append(reg_loss)
            exp3ix_regret.append(b)
        
        R_eta.append(sum(R)/50)
    print R_eta
    eta = []
    regret_mean = []
    regret_err = []
    freedom_degree = len(exp3ix_regret[0]) - 2
    for regret in exp3ix_regret:
        eta.append(regret[0])
        regret_mean.append(np.mean(regret[1:]))
        regret_err.append(ss.t.ppf(0.95, freedom_degree) * ss.sem(regret[1:]))
    colors = list("rgbcmyk")
    shape =['--^', '--d', '--v']
    plt.errorbar(eta, regret_mean, regret_err, color=colors[2])
    plt.plot(eta, regret_mean, colors[2] + shape[2], label='EXP3-IX')
    print "EXP3-IX Done!"



main1()
main2()
main3()

plt.legend(loc='upper right', numpoints=1)
plt.title("Pseudo Regret vs Learning Rate for T = 10^5 and 50 Sample paths")
plt.xlabel("Learning Rate")
plt.ylabel("Pseudo Regret")
plt.savefig("Q2.png", bbox_inches='tight')
plt.close()



