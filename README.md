# oneunit
Domain Specific Language (DSL) for Système international d'unités (SI) arithmetic

## SI units

### Common Derived Units and Their 7D Vectors

We use the standard order of the 7 SI base units as coordinates:

```math

\bigl[L,\; M,\; T,\; I,\; \Theta,\; N,\; J\bigr]

```
where  
- $$\(L\)$$ = length (meter, m)  
- $$\(M\)$$ = mass (kilogram, kg)  
- $$\(T\)$$ = time (second, s)  
- $$\(I\)$$ = electric current (ampere, A)  
- $$\(\Theta\)$$ = thermodynamic temperature (kelvin, K)  
- $$\(N\)$$ = amount of substance (mole, mol)  
- $$\(J\)$$ = luminous intensity (candela, cd)  

In the vectors below, any unmentioned base units have exponent 0 (e.g., temperature, amount of substance, or luminous intensity). For brevity, we typically focus on mechanical/electrical dimensions:

#### Mechanics & Thermodynamics

| Unit             | Symbol | Dimensions in Base Units                     | 7D Vector         |
|------------------|--------|----------------------------------------------|-------------------|
| **Frequency**    | Hz     | $$\(T^{-1}\)$$                                   | $$\((0,0,-1,0,0,0,0)\)$$ |
| **Force**        | N      | $$\(M^1 L^1 T^{-2}\)$$                           | $$\((1,1,-2,0,0,0,0)\)$$ |
| **Pressure**     | Pa     | $$\(\tfrac{F}{A} = M^1 L^{-1} T^{-2}\)$$         | $$\((-1,1,-2,0,0,0,0)\)$$ |
| **Energy/Work**  | J      | $$\(M^1 L^2 T^{-2}\)$$                           | $$\((2,1,-2,0,0,0,0)\)$$ |
| **Power**        | W      | $$\(M^1 L^2 T^{-3}\)$$                           | $$\((2,1,-3,0,0,0,0)\)$$ |
| **Velocity**     | m/s    | $$\(L^1 T^{-1}\)$$                               | $$\((1,0,-1,0,0,0,0)\)$$ |
| **Acceleration** | m/s²   | $$\(L^1 T^{-2}\)$$                               | $$\((1,0,-2,0,0,0,0)\)$$ |
| **Action**       | J·s    | $$\(M^1 L^2 T^{-1}\)$$                           | $$\((2,1,-1,0,0,0,0)\)$$ |
| **Dynamic Viscosity** | Pa·s | $$\(M^1 L^{-1} T^{-1}\)$$                    | $$\((-1,1,-1,0,0,0,0)\)$$ |
| **Kinematic Viscosity** | m²/s | $$\(L^2 T^{-1}\)$$                         | $$\((2,0,-1,0,0,0,0)\)$$ |


#### Electricity & Magnetism

| Unit               | Symbol | Dimensions in Base Units                                    | 7D Vector          |
|--------------------|--------|-------------------------------------------------------------|--------------------|
| **Charge**         | C      | $$\(T^1 I^1\)$$ (ampere × second)                              | $$\((0,0,1,1,0,0,0)\)$$  |
| **Electric Potential** | V  | $$\(M^1 L^2 T^{-3} I^{-1}\)$$ (joule per coulomb)              | $$\((2,1,-3,-1,0,0,0)\)$$ |
| **Capacitance**    | F      | $$\(M^{-1} L^{-2} T^4 I^2\)$$                                   | $$\((-2,-1,4,2,0,0,0)\)$$  |
| **Resistance**     | Ω      | $$\(M^1 L^2 T^{-3} I^{-2}\)$$                                   | $$\((2,1,-3,-2,0,0,0)\)$$ |
| **Conductance**    | S      | $$\(\Omega^{-1} = M^{-1} L^{-2} T^3 I^2\)$$                     | $$\((-2,-1,3,2,0,0,0)\)$$  |
| **Magnetic Flux**  | Wb     | $$\(M^1 L^2 T^{-2} I^{-1}\)$$                                   | $$\((2,1,-2,-1,0,0,0)\)$$ |
| **Magnetic Flux Density** | T (tesla) | $$\(\frac{\text{weber}}{\text{m}^2} = M^1 T^{-2} I^{-1}\)$$ | $$\((0,1,-2,-1,0,0,0)\)$$ |
| **Inductance**     | H      | $$\(M^1 L^2 T^{-2} I^{-2}\)$$                                   | $$\((2,1,-2,-2,0,0,0)\)$$ |

#### Additional Examples (Radiation, Etc.)

| Unit             | Symbol | Dimensions (subset)                  | 7D Vector (partial)  |
|------------------|--------|--------------------------------------|-----------------------|
| **Becquerel**    | Bq     | $$\(T^{-1}\) (1/s)$$                     | $$\((0,0,-1,0,0,0,0)\)$$  |
| **Gray**         | Gy     | $$\( \frac{\text{J}}{\text{kg}} = L^2 T^{-2}\)$$ | $$\((2,0,-2,0,0,0,0)\)$$  |
| **Sievert**      | Sv     | same as Gray ($$\(L^2 T^{-2}\)$$)        | $$\((2,0,-2,0,0,0,0)\)$$  |
| **Katal**        | kat    | $$\(\tfrac{\text{mol}}{\text{s}} = N^1 T^{-1}\)$$ | $$\((0,0,-1,0,0,1,0)\)$$ |

