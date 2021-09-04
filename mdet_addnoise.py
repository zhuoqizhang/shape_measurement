import galsim
import numpy as np
import ngmix
import tqdm


def sim_func(gal, psf, g1, g2, noise, n_sigma=1e-4):
    
    jacobian=ngmix.DiagonalJacobian(scale=0.2, row=26, col=26)
        
    obj =  galsim.Convolve(gal.shear(g1=g1, g2=g2), psf)
    obj_im = obj.drawImage(nx=53, ny=53, scale=0.2)
    psf_im = psf.drawImage(nx=53, ny=53, scale=0.2)
    
    psf_obs = ngmix.Observation(
        image=psf_im.array,
        jacobian = jacobian,
    )
    
    obj_obs = ngmix.Observation(
        image=obj_im.array + noise,
        weight=np.ones((53,53))/n_sigma**2,
        psf=psf_obs,
        jacobian = jacobian,
    )
    
    return obj_obs

def add_noise(obj_obs_deep, noise_deep2wide, n_sigma_deep2wide):
    
    image = obj_obs_deep.image
    weight = obj_obs_deep.weight
    psf_obs = obj_obs_deep.psf
    jacobian = obj_obs_deep.jacobian
    
    obj_obs_wide = ngmix.Observation(
        image=image+noise_deep2wide,
        weight=np.ones((53,53))/(1/weight+n_sigma_deep2wide**2),
        psf=psf_obs,
        jacobian = jacobian
    )
    
    return obj_obs_wide

def make_ngmix_obs(array, psf, n_sigma):
    
    jacobian=ngmix.DiagonalJacobian(scale=0.2, row=26, col=26)
    psf_im = psf.drawImage(nx=53, ny=53, scale=0.2)

    psf_obs = ngmix.Observation(
        image=psf_im.array,
        jacobian = jacobian,
    )
    
    obj_obs = ngmix.Observation(
        image = array,
        weight=np.ones((53,53))/n_sigma**2,
        psf=psf_obs,
        jacobian = jacobian,
    )
    return obj_obs

def two_stage(gal, psf, blank, rng_noise, rng_metacal, h=0.01, n_mu=0.0, n_sigma_wide=1e-4, n_sigma_deep=1e-6, weight_fwhm=1.2):
    
    # make noises
    noise_wide = rng_noise.normal(n_mu, n_sigma_wide, (53, 53))
    noise_deep = rng_noise.normal(n_mu, n_sigma_deep, (53, 53))
    
    
    # make observations
    obs_deep = sim_func(gal=gal, psf=psf, g1=0.0, g2=0.0, noise=noise_deep, n_sigma=n_sigma_deep)
    obs_wide = sim_func(gal=gal, psf=psf, g1=0.0, g2=0.0, noise=noise_wide, n_sigma=n_sigma_wide)
    deepnoise = sim_func(gal=blank, psf=psf, g1=0.0, g2=0.0, noise=noise_deep, n_sigma=n_sigma_deep)

    
    # metacal the blank images, for adding noise
    metacal_deepnoise = ngmix.metacal.get_all_metacal(deepnoise, psf='fitgauss', fixnoise=False,
                                               step=h, rng=rng_metacal)
    noshear_deepnoise = metacal_deepnoise['noshear']    
    noshear_deepnoise_im = noshear_deepnoise.image.copy()
    noshear_deepnoise_std = np.std(noshear_deepnoise_im)
    
    
    # add noshear_deepnoise to wide observation
    obs_addnoise = add_noise(obs_wide, noshear_deepnoise_im, noshear_deepnoise_std)
    
    

    # metacal the deep images
    metacal_ims = ngmix.metacal.get_all_metacal(obs_deep, psf='fitgauss', fixnoise=True,
                                               step=h, rng=rng_metacal)
    noshear_deep = metacal_ims['noshear']
    shearg1p_deep = metacal_ims['1p']
    shearg1m_deep = metacal_ims['1m']
    shearg2p_deep = metacal_ims['2p']
    shearg2m_deep = metacal_ims['2m']
    
    shearg1p_addnoise = add_noise(shearg1p_deep, noise_wide, n_sigma_wide)
    shearg1m_addnoise = add_noise(shearg1m_deep, noise_wide, n_sigma_wide)
    shearg2p_addnoise = add_noise(shearg2p_deep, noise_wide, n_sigma_wide)
    shearg2m_addnoise = add_noise(shearg2m_deep, noise_wide, n_sigma_wide)
    
    
    # now measure response
    from ngmix.ksigmamom import KSigmaMom
    ks = KSigmaMom(2.0) # 2.0
    noshear_meas = ks.go(obs_addnoise)
    shearg1p_meas = ks.go(shearg1p_addnoise)
    shearg1m_meas = ks.go(shearg1m_addnoise)
    shearg2p_meas = ks.go(shearg2p_addnoise)
    shearg2m_meas = ks.go(shearg2m_addnoise)
    
    # make selection
    flag_noshear = noshear_meas['flags']
    flag_g1p = shearg1p_meas['flags']
    flag_g1m = shearg1m_meas['flags']
    flag_g2p = shearg2p_meas['flags']
    flag_g2m = shearg2m_meas['flags']
    
    if flag_noshear==0 and flag_g1p==0 and flag_g1m==0 and flag_g2p==0 and flag_g2m==0:
        flag = 0
    else: 
        flag = 1
    
    s2n_noshear = noshear_meas['s2n']
    s2n_g1p = shearg1p_meas['s2n']
    s2n_g1m = shearg1m_meas['s2n']
    s2n_g2p = shearg2p_meas['s2n']
    s2n_g2m = shearg2m_meas['s2n']
    
    if s2n_noshear>5 and s2n_g1p>5 and s2n_g1m>5 and s2n_g2p>5 and s2n_g2m>5: 
        s2n_flag = 0
    else: 
        s2n_flag = 1
        
    t_noshear = noshear_meas['T']
    t_g1p = shearg1p_meas['T']
    t_g1m = shearg1m_meas['T']
    t_g2p = shearg2p_meas['T']
    t_g2m = shearg2m_meas['T']
        
    if t_noshear>0.2 and t_g1p>0.2 and t_g1m>0.2 and t_g2p>0.2 and t_g2m>0.2: 
        t_flag = 0
    else: 
        t_flag = 1
    
    e_noshear = np.max(np.abs(noshear_meas['e']))
    e_g1p = np.max(np.abs(shearg1p_meas['e']))
    e_g1m = np.max(np.abs(shearg1m_meas['e']))
    e_g2p = np.max(np.abs(shearg2p_meas['e']))
    e_g2m = np.max(np.abs(shearg2m_meas['e']))
    
    if e_noshear<1 and e_g1p<1 and e_g1m<1 and e_g2p<1 and e_g2m<1: 
        e_flag = 0
    else: 
        e_flag = 1
    
   
    r1 = (shearg1p_meas['e1']-shearg1m_meas['e1'])/(2*h)
    r2 = (shearg2p_meas['e2']-shearg2m_meas['e2'])/(2*h)
    
    if flag==0 and s2n_flag==0 and t_flag==0 and e_flag==0: 
        discard = 0
    else: 
        discard = 1
    
    
    return noshear_meas['e'], np.array([r1,r2]), discard

def worker(task):
    seed1, seed2 = task
    #print(task)
    e_arr = []
    r_arr = []

    rng_noise = np.random.RandomState(seed1)
    rng_metacal = np.random.RandomState(seed2)
    e_p, r_p, flag_p = two_stage(gal_p, psf, blank, rng_noise, rng_metacal, n_sigma_wide=5e-3, n_sigma_deep=1.5e-3)
    e_m, r_m, flag_m = two_stage(gal_m, psf, blank, rng_noise, rng_metacal,n_sigma_wide=5e-3, n_sigma_deep=1.5e-3)
    if flag_p==0 and flag_m==0:
        e = [(e_p[0]-e_m[0])/2., (e_p[1]+e_m[1])/2.]
        return (e, (r_p+r_m)/2.)
    else: 
        pass

def main(pool):
    # Here we generate some fake data
    import random
    
    n = 1000000
    a = [random.randint(0,2**31-1) for _ in range(n)]
    b = [random.randint(0,2**31-1) for _ in range(n)]

    tasks = list(zip(a, b))
    results = pool.map(worker, tasks)
    pool.close()
    
    return results

gal_p = galsim.Exponential(half_light_radius=0.5).shear(g1=0.02, g2=0)
gal_m = galsim.Exponential(half_light_radius=0.5).shear(g1=-0.02, g2=0)
blank = galsim.Exponential(half_light_radius=0.5,flux=0.0).shear(g1=-0.02, g2=0)
psf = galsim.Gaussian(fwhm=0.9)



if __name__ == "__main__":
    import sys
    from schwimmbad import MPIPool

    pool = MPIPool()

    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    results = main(pool)
    
    e_arr = []
    r_arr = []
    for i in range(len(results)):
        if results[i] is not None: 
            e_arr.append(results[i][0])
            r_arr.append(results[i][1])
    e_arr = np.array(e_arr)
    r_arr = np.array(r_arr)
    
    print(len(e_arr))
    
    np.save('run8/e_arr',e_arr)
    np.save('run8/r_arr',r_arr)