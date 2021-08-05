#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;


// Create only once the default random engine
static default_random_engine gen;


// Particle filter initialization.
// Set number of particles and initialize them to first position based on GPS estimate.
void ParticleFilter::init(double x, double y, double theta, double std[]) {

    /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  	num_particles = 500;  // TODO: Set the number of particle

    // Creates normal (Gaussian) distribution for p_x, p_y and p_theta
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; ++i) {

        // Instantiate a new particle
        Particle part;
        part.id		= int(i);
        part.weight	= 1.0;
        part.x			= dist_x(gen);
        part.y			= dist_y(gen);
        part.theta		= dist_theta(gen);

        // Add the particle to the particle filter set
        particles.push_back(part);
    }

    is_initialized = true;

}


// Move each particle according to bicycle motion model (taking noise into account)
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
   /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   *  xf​=x0​+v/θ˙​[sin(θ0​+θ˙(dt))−sin(θ0​)]
   *  yf​=y0​+θ˙v​[cos(θ0​)−cos(θ0​+θ˙(dt))]
   *  θf​=θ0​+θ˙(dt)
   */

    for (int i = 0; i < num_particles; ++i) {

        // Gather old values
        double x	 = particles[i].x;
        double y	 = particles[i].y;
        double theta = particles[i].theta;


        if (abs(yaw_rate) > 1e-5) {
            // Apply equations of motion model (turning)
            
            x	   = x + velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
            y	   = y + velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
            theta = theta + yaw_rate * delta_t;
        } else {
            // Apply equations of motion model (going straight)
//             theta = theta;
            x	   = x + velocity * delta_t * cos(theta);
            y	   = y + velocity * delta_t * sin(theta);
        }

        // Initialize normal distributions centered on predicted values
        normal_distribution<double> dist_x(x, std_pos[0]);
        normal_distribution<double> dist_y(y, std_pos[1]);
        normal_distribution<double> dist_theta(theta, std_pos[2]);

        // Update particle with noisy prediction
        particles[i].x	   = dist_x(gen);
        particles[i].y	   = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }
}


// Finds which observations correspond to which landmarks by using a nearest-neighbor data association
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations){
    /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
 
    for (auto& obs : observations) {
        double min = numeric_limits<double>::max();

        for (const auto& pred : predicted) {
            double d = dist(obs.x, obs.y, pred.x, pred.y);
            if (d < min) {
                obs.id	 = pred.id;
                min = d;
            }
        }
    }
}


// Update the weight of each particle taking into account current measurements.
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                     					const std::vector<LandmarkObs> &observations,
                     					const Map &map_landmarks) {
    /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   *   xm​=xp​+(cosθ×xc​)−(sinθ×yc​)

	*  ym​=yp​+(sinθ×xc​)+(cosθ×yc​)
    *	
   */
  

    // Gather std values for readability
    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
  	double norm_factor = 0.0;

    // Iterate over all particles
    for (int i = 0; i < num_particles; ++i) {

        // Gather current particle values
        double p_x	   = particles[i].x;
        double p_y	   = particles[i].y;
        double p_theta = particles[i].theta;

        // List all landmarks within sensor range
        vector<LandmarkObs> predicted_landmarks;

        for (const auto& map_landmark : map_landmarks.landmark_list) {
            int id   = map_landmark.id_i;
            double x = (double) map_landmark.x_f;
            double y = (double) map_landmark.y_f;

            double d = dist(p_x, p_y, x, y);
            if (d < sensor_range) {
                LandmarkObs pred;
                pred.id = id;
                pred.x = x;
                pred.y = y;
                predicted_landmarks.push_back(pred);
            }
        }

        // List all observations in map coordinates
        vector<LandmarkObs> observed_landmarks_map;
        for (size_t j = 0; j < observations.size(); ++j) {

            // Convert observation from particle(vehicle) to map coordinate system
            LandmarkObs obs;
            obs.x = cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y + p_x;
            obs.y = sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y + p_y;

            observed_landmarks_map.push_back(obs);
        }

        // Find which observations correspond to which landmarks (associate ids)
        dataAssociation(predicted_landmarks, observed_landmarks_map);

        // Compute the likelihood for each particle, that is the probablity of obtaining
        // current observations being in state (particle_x, particle_y, particle_theta)
        double probs = 1.0;

        double pred_x, pred_y;
        for (const auto& obs : observed_landmarks_map) {

            // Find corresponding landmark on map for centering gaussian distribution
            for (const auto& landmark: predicted_landmarks)
                if (obs.id == landmark.id) {
                    pred_x = landmark.x;
                    pred_y = landmark.y;
                    break;
                }

            double norm_factor = 2 * M_PI * std_x * std_y;
            double prob = exp( -( pow(obs.x - pred_x, 2) / (2 * std_x * std_x) + pow(obs.y - pred_y, 2) / (2 * std_y * std_y) ) );

            probs *= prob / norm_factor;
        }

        particles[i].weight = probs;
        norm_factor += probs;

    } // end loop for each particle

    // Normalize weights so they sum to one
    for (auto& particle : particles)
        particle.weight /= (norm_factor + numeric_limits<double>::epsilon());
}



void ParticleFilter::resample() {
   /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
    vector<double> particle_weights;
    for (const auto& particle : particles)
        particle_weights.push_back(particle.weight);

    discrete_distribution<int> weighted_distribution(particle_weights.begin(), particle_weights.end());

    vector<Particle> selected;
    for (int i = 0; i < num_particles; ++i) {
        int index = weighted_distribution(gen);
        selected.push_back(particles[index]);
    }

    particles = selected;

}


void  ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                       const std::vector<double>& sense_x, 
                       const std::vector<double>& sense_y)
{
 // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

}


string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}


string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
