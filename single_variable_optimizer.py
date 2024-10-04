import random

class SingleVariableOptimizer:

    def __init__(self, objective_function, a, b, n_steps=20, e=0.01, minimize:bool=True):
        self.objective_function = objective_function # Objective Function 
        self.a = float(a) # Lower bound of the range 
        self.b = float(b) # Upper bound of the range
        step = (b-a)/n_steps # Size of each step based on n_steps i.e number of steps
        self.step_size = float(step)
        self.e = float(e) # Value of 'e' for Newton Raphson Method
        self.minimize = minimize # Boolean: True if objective is to minimize, False if objective is to maximize
        self.results = []
        self.logger = []
        self.x_range = []
        self.call = 0

    # Wrapper method to count calls to the objective function
    def call_objective_function(self, x):
        self.call += 1
        return self.objective_function(x)

    def bounding_phase_method(self):
        self.logger.append(f"INITIATING BOUNDING PHASE METHOD, range = ({self.decimal(self.a), self.decimal(self.b)})")
        # Step 1: Choose an initial guess x0 and set k = 0
        x0 = 0.5*(self.a + self.b)
        self.logger.append(f"Initial random guess = {self.decimal(x0)}")
        k = 0
        
        # MINIMIZE
        # Step 2: Determine the direction of delta
        if self.minimize:
            delta = self.step_size
            if self.call_objective_function(x0 - abs(delta)) >= self.call_objective_function(x0) >= self.call_objective_function(x0 + abs(delta)):
                delta = abs(delta)  # Delta is positive
            elif self.call_objective_function(x0 - abs(delta)) <= self.call_objective_function(x0) <= self.call_objective_function(x0 + abs(delta)):
                delta = -abs(delta)  # Delta is negative
            else:
                return (self.a, self.b)  # Go back to step 1
            
            # Step 3 & 4: Expand the search
            while True:
                self.logger.append(f"Iteration {k+1} : x = {self.decimal(x0)}, delta = {self.decimal(delta)}")  
                self.x_range.append(self.decimal(x0))              
                x_next = x0 + (2**k) * delta
                if self.call_objective_function(x_next) < self.call_objective_function(x0):
                    k += 1
                    x0 = x_next  # Update x0
                else:
                    x_prev = x0 - (2**(k-1)) * delta
                    self.logger.append(f"Final range from Bounding Phase: {(self.decimal(min(x_prev, x_next)), self.decimal(max(x_prev, x_next)))}")
                    return (min(x_prev, x_next), max(x_prev, x_next))  # Return the interval
            
    def newton_raphson_method(self, c, d):
        self.logger.append(f"INITIATING NEWTON RAPHSON METHOD, range = ({self.decimal(c), self.decimal(d)})")

        # Step 1: Choose an initial guess x1 and set k = 1
        x1 = random.uniform(c, d)
        self.logger.append(f"Initial random guess = {self.decimal(x1)}")
        k = 1

        while True:
            self.logger.append(f"Iteration {k} : x = {self.decimal(x1)}")
            self.x_range.append(self.decimal(x1))
            # Compute f'(xk) and f''(xk)
            f_prime = self.derivative(x1)
            f_double_prime = self.second_derivative(x1)

            # Check for zero or near-zero second derivative
            if abs(f_double_prime) < 1e-10:
                f_double_prime = 1e-10

            # Step 3: Calculate xk+1
            x_next = x1 - f_prime / f_double_prime

            # Step 4: Check termination condition
            if abs(self.derivative(x_next)) < self.e or k>100:
                return (x_next, self.call_objective_function(x_next))

            # Update x1 and k
            x1 = x_next
            k += 1

    # Helper Function for derivative calculation
    def derivative(self, x, h=1e-5):
        h = float(h)
        return (self.call_objective_function(x + h) - self.call_objective_function(x - h)) / (2 * h)

    # Helper Function for double derivative calculation
    def second_derivative(self, x, h=1e-5):
        h = float(h)
        return (self.call_objective_function(x + h) - 2 * self.call_objective_function(x) + self.call_objective_function(x - h)) / (h**2)

    def optimize(self):
        # Step 1: Perform bounding phase method to get a new range (x, y)
        x, y = self.bounding_phase_method()
        
        # Step 2: Perform Newton-Raphson method to find the optimized value
        result = self.newton_raphson_method(x, y)

        # Final Result:
        self.logger.append(f"->Final result : {result}")
        return result
    
    # Helper Function to round decimals
    def decimal(self, x):
        return round(x,3)        

    # Helper Function to log the iterations 
    def log(self):
        return '\n'.join(self.logger)
    