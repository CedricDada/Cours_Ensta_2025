
#include <string>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <thread>
#include <chrono>
#include <mpi.h>
#include "model.hpp"
#include "display.hpp"

using namespace std::string_literals;
using namespace std::chrono_literals;

struct ParamsType
{
    double length{1.};
    unsigned discretization{20u};
    std::array<double,2> wind{0.,0.};
    Model::LexicoIndices start{10u,10u};
};

void analyze_arg( int nargs, char* args[], ParamsType& params )
{
    if (nargs ==0) return;
    std::string key(args[0]);
    if (key == "-l"s)
    {
        if (nargs < 2)
        {
            std::cerr << "Manque une valeur pour la longueur du terrain !" << std::endl;
            exit(EXIT_FAILURE);
        }
        params.length = std::stoul(args[1]);
        analyze_arg(nargs-2, &args[2], params);
        return;
    }
    auto pos = key.find("--longueur=");
    if (pos < key.size())
    {
        auto subkey = std::string(key,pos+11);
        params.length = std::stoul(subkey);
        analyze_arg(nargs-1, &args[1], params);
        return;
    }

    if (key == "-n"s)
    {
        if (nargs < 2)
        {
            std::cerr << "Manque une valeur pour le nombre de cases par direction pour la discrétisation du terrain !" << std::endl;
            exit(EXIT_FAILURE);
        }
        params.discretization = std::stoul(args[1]);
        analyze_arg(nargs-2, &args[2], params);
        return;
    }
    pos = key.find("--number_of_cases=");
    if (pos < key.size())
    {
        auto subkey = std::string(key, pos+18);
        params.discretization = std::stoul(subkey);
        analyze_arg(nargs-1, &args[1], params);
        return;
    }

    if (key == "-w"s)
    {
        if (nargs < 2)
        {
            std::cerr << "Manque une paire de valeurs pour la direction du vent !" << std::endl;
            exit(EXIT_FAILURE);
        }
        std::string values =std::string(args[1]);
        params.wind[0] = std::stod(values);
        auto pos = values.find(",");
        if (pos == values.size())
        {
            std::cerr << "Doit fournir deux valeurs séparées par une virgule pour définir la vitesse" << std::endl;
            exit(EXIT_FAILURE);
        }
        auto second_value = std::string(values, pos+1);
        params.wind[1] = std::stod(second_value);
        analyze_arg(nargs-2, &args[2], params);
        return;
    }
    pos = key.find("--wind=");
    if (pos < key.size())
    {
        auto subkey = std::string(key, pos+7);
        params.wind[0] = std::stoul(subkey);
        auto pos = subkey.find(",");
        if (pos == subkey.size())
        {
            std::cerr << "Doit fournir deux valeurs séparées par une virgule pour définir la vitesse" << std::endl;
            exit(EXIT_FAILURE);
        }
        auto second_value = std::string(subkey, pos+1);
        params.wind[1] = std::stod(second_value);
        analyze_arg(nargs-1, &args[1], params);
        return;
    }

    if (key == "-s"s)
    {
        if (nargs < 2)
        {
            std::cerr << "Manque une paire de valeurs pour la position du foyer initial !" << std::endl;
            exit(EXIT_FAILURE);
        }
        std::string values =std::string(args[1]);
        params.start.column = std::stod(values);
        auto pos = values.find(",");
        if (pos == values.size())
        {
            std::cerr << "Doit fournir deux valeurs séparées par une virgule pour définir la position du foyer initial" << std::endl;
            exit(EXIT_FAILURE);
        }
        auto second_value = std::string(values, pos+1);
        params.start.row = std::stod(second_value);
        analyze_arg(nargs-2, &args[2], params);
        return;
    }
    pos = key.find("--start=");
    if (pos < key.size())
    {
        auto subkey = std::string(key, pos+8);
        params.start.column = std::stoul(subkey);
        auto pos = subkey.find(",");
        if (pos == subkey.size())
        {
            std::cerr << "Doit fournir deux valeurs séparées par une virgule pour définir la vitesse" << std::endl;
            exit(EXIT_FAILURE);
        }
        auto second_value = std::string(subkey, pos+1);
        params.start.row = std::stod(second_value);
        analyze_arg(nargs-1, &args[1], params);
        return;
    }
}

ParamsType parse_arguments( int nargs, char* args[] )
{
    if (nargs == 0) return {};
    if ( (std::string(args[0]) == "--help"s) || (std::string(args[0]) == "-h") )
    {
        std::cout << 
R"RAW(Usage : simulation [option(s)]
  Lance la simulation d'incendie en prenant en compte les [option(s)].
  Les options sont :
    -l, --longueur=LONGUEUR     Définit la taille LONGUEUR (réel en km) du carré représentant la carte de la végétation.
    -n, --number_of_cases=N     Nombre n de cases par direction pour la discrétisation
    -w, --wind=VX,VY            Définit le vecteur vitesse du vent (pas de vent par défaut).
    -s, --start=COL,ROW         Définit les indices I,J de la case où commence l'incendie (milieu de la carte par défaut)
)RAW";
        exit(EXIT_SUCCESS);
    }
    ParamsType params;
    analyze_arg(nargs, args, params);
    return params;
}

bool check_params(ParamsType& params)
{
    bool flag = true;
    if (params.length <= 0)
    {
        std::cerr << "[ERREUR FATALE] La longueur du terrain doit être positive et non nulle !" << std::endl;
        flag = false;
    }

    if (params.discretization <= 0)
    {
        std::cerr << "[ERREUR FATALE] Le nombre de cellules par direction doit être positive et non nulle !" << std::endl;
        flag = false;
    }

    if ( (params.start.row >= params.discretization) || (params.start.column >= params.discretization) )
    {
        std::cerr << "[ERREUR FATALE] Mauvais indices pour la position initiale du foyer" << std::endl;
        flag = false;
    }
    
    return flag;
}

void display_params(ParamsType const& params)
{
    std::cout << "Parametres définis pour la simulation : \n"
              << "\tTaille du terrain : " << params.length << std::endl 
              << "\tNombre de cellules par direction : " << params.discretization << std::endl 
              << "\tVecteur vitesse : [" << params.wind[0] << ", " << params.wind[1] << "]" << std::endl
              << "\tPosition initiale du foyer (col, ligne) : " << params.start.column << ", " << params.start.row << std::endl;
}
// Fonction utilitaire pour afficher un message de log avec le rang dans le sous-communicateur
// Fonction utilitaire pour afficher un message de log avec le rang dans le sous-communicateur
void log_message(MPI_Comm comm, const std::string &msg) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    std::cout << "[Rank " << rank << "] " << msg << std::endl;
}

// Fonction mise à jour pour les cellules fantômes avec réordonnement des appels
void update_ghost_cells(std::vector<std::uint8_t>& local_fire,
                          int local_rows, int cols,
                          MPI_Comm newComm, int comp_rank, int comp_size)
{
    MPI_Request requests[2];
    int req_count = 0;
    
    log_message(newComm, "Début de l'update des cellules fantômes");
    
    // Communication vers le haut (réception de la ligne fantôme supérieure)
    // Le voisin inférieur (comp_rank - 1) envoie sa première ligne vers le bas (tag 1)
    if (comp_rank > 0) {
        log_message(newComm, "Poste réception de la ligne fantôme supérieure du voisin " + std::to_string(comp_rank-1));
        MPI_Irecv(&local_fire[0], cols, MPI_UINT8_T, comp_rank - 1, 1, newComm, &requests[req_count++]);
    }
    // Communication vers le bas (réception de la ligne fantôme inférieure)
    // Le voisin supérieur (comp_rank + 1) envoie sa dernière ligne vers le haut (tag 0)
    if (comp_rank < comp_size - 1) {
        log_message(newComm, "Poste réception de la ligne fantôme inférieure du voisin " + std::to_string(comp_rank+1));
        MPI_Irecv(&local_fire[(local_rows + 1) * cols], cols, MPI_UINT8_T, comp_rank + 1, 0, newComm, &requests[req_count++]);
    }
    
    // Envoi de la première ligne locale vers le bas (destiné à être la ligne fantôme supérieure du voisin inférieur)
    if (comp_rank > 0) {
        log_message(newComm, "Envoi de la première ligne locale (ligne 1) au voisin supérieur " + std::to_string(comp_rank-1));
        MPI_Ssend(&local_fire[cols], cols, MPI_UINT8_T, comp_rank - 1, 0, newComm);
    }
    // Envoi de la dernière ligne locale vers le haut (destiné à être la ligne fantôme inférieure du voisin supérieur)
    if (comp_rank < comp_size - 1) {
        log_message(newComm, "Envoi de la dernière ligne locale (ligne " + std::to_string(local_rows) + ") au voisin inférieur " + std::to_string(comp_rank+1));
        MPI_Ssend(&local_fire[local_rows * cols], cols, MPI_UINT8_T, comp_rank + 1, 1, newComm);
    }
    
    if (req_count > 0) {
        log_message(newComm, "Attente de la fin des réceptions des cellules fantômes");
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    }
    log_message(newComm, "Fin de l'update des cellules fantômes");
}
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    // Duplication du communicateur global
    MPI_Comm globComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &globComm);

    int world_rank, world_size;
    MPI_Comm_rank(globComm, &world_rank);
    MPI_Comm_size(globComm, &world_size);

    // Création du sous-communicateur : couleur 0 pour l'affichage (global rank 0), couleur 1 pour les calculs
    int color = (world_rank == 0 ? 0 : 1);
    MPI_Comm newComm;
    MPI_Comm_split(globComm, color, world_rank, &newComm);

    // Récupération des paramètres
    auto params = parse_arguments(argc - 1, &argv[1]);

    // ------------------------ Processus d'affichage (global rank 0) ------------------------
    if (world_rank == 0)
    {
        display_params(params);
        auto displayer = Displayer::init_instance(params.discretization, params.discretization);
        bool keep_running = true;
        const int tag_signal = 0, tag_veg = 1, tag_fire = 2;

        while (keep_running)
        {
            // Signaler au processus de calcul principal (global rank 1) qu'on est prêt à recevoir
            int signal = 1;
            MPI_Send(&signal, 1, MPI_INT, 1, tag_signal, globComm);

            // Réception de la grille globale (les tableaux complets)
            std::vector<std::uint8_t> global_vegetation(params.discretization * params.discretization);
            std::vector<std::uint8_t> global_fire(params.discretization * params.discretization);
            MPI_Recv(global_vegetation.data(), global_vegetation.size(), MPI_UINT8_T, 1, tag_veg, globComm, MPI_STATUS_IGNORE);
            MPI_Recv(global_fire.data(), global_fire.size(), MPI_UINT8_T, 1, tag_fire, globComm, MPI_STATUS_IGNORE);

            // Mise à jour de l'affichage via SDL
            displayer->update(global_vegetation, global_fire);

            // Gestion d'événement (par exemple fermeture de la fenêtre)
            SDL_Event event;
            if (SDL_PollEvent(&event) && event.type == SDL_QUIT)
            {
                // Envoi d'un signal d'arrêt (valeur -1) au processus de calcul principal
                signal = -1;
                MPI_Send(&signal, 1, MPI_INT, 1, tag_signal, globComm);
                keep_running = false;
            }
        }
    }
    // --- Partie calcul (dans le else du global_rank != 0) ---
    else // Processus de calcul
    {
        // Récupération du rang dans le sous-communicateur de calcul
        int comp_rank, comp_size;
        MPI_Comm_rank(newComm, &comp_rank);
        MPI_Comm_size(newComm, &comp_size);

        // Pour la décomposition, supposons ici une répartition par tranches horizontales
        int total_rows = params.discretization;
        int rows_per_proc = total_rows / comp_size;
        int remainder = total_rows % comp_size;
        int local_rows = rows_per_proc + (comp_rank < remainder ? 1 : 0);
        int cols = params.discretization;

        // Allocation de la portion locale incluant 2 lignes fantômes (en haut et en bas)
        int local_rows_with_ghosts = local_rows + 2;
        std::vector<std::uint8_t> local_vegetation(local_rows_with_ghosts * cols, 255);
        std::vector<std::uint8_t> local_fire(local_rows_with_ghosts * cols, 0);

        // Déterminer quel processus contient la ligne de départ
        int global_start_row = params.start.row;
        int owner_rank = 0, accumulated_rows = 0;
        for (int i = 0; i < comp_size; ++i) {
            int r = rows_per_proc + (i < remainder ? 1 : 0);
            if (global_start_row >= accumulated_rows && global_start_row < accumulated_rows + r) {
                owner_rank = i;
                break;
            }
            accumulated_rows += r;
        }

        // Initialiser le feu localement si c'est le bon processus
        if (comp_rank == owner_rank) {
            int local_start_row = global_start_row - accumulated_rows + 1; // +1 pour ignorer la fantôme
            size_t index = local_start_row * cols + params.start.column;
            local_fire[index] = 255;
        }

        // Initialisation du modèle local
        Model local_model(
            params.length, 
            params.discretization, 
            params.wind, 
            params.start,
            local_rows, 
            accumulated_rows
        );

        bool simulation_running = true;
        const int tag_signal = 0, tag_veg = 1, tag_fire = 2;

        while (simulation_running)
        {
            // Échange des cellules fantômes
            update_ghost_cells(local_fire, local_rows, cols, newComm, comp_rank, comp_size);

            // Mise à jour locale de la simulation
            simulation_running = local_model.update();

            // Rassemblement des données avec MPI_Gatherv
            std::vector<std::uint8_t> global_vegetation, global_fire;
            std::vector<int> counts(comp_size), displs(comp_size);
            int local_count = local_rows * cols;

            int offset = 0;
            for (int i = 0; i < comp_size; ++i) {
                int rows = rows_per_proc + (i < remainder ? 1 : 0);
                counts[i] = rows * cols;
                displs[i] = offset;
                offset += rows * cols;
            }
            if (comp_rank == 0) {
                global_vegetation.resize(total_rows * cols);
                global_fire.resize(total_rows * cols);
            }

            auto local_veg = local_model.vegetal_map(); // Extraction des données locales
            auto local_f = local_model.fire_map();      // Extraction des données locales

            MPI_Gatherv(local_veg.data(), local_count, MPI_UINT8_T,
                        global_vegetation.data(), counts.data(), displs.data(), MPI_UINT8_T,
                        0, newComm);
            MPI_Gatherv(local_f.data(), local_count, MPI_UINT8_T,
                        global_fire.data(), counts.data(), displs.data(), MPI_UINT8_T,
                        0, newComm);

            // Communication avec le processus d'affichage
            if (comp_rank == 0) {
                int flag = 0;
                MPI_Status status;
                MPI_Iprobe(0, tag_signal, globComm, &flag, &status);
                if (flag) {
                    int signal;
                    MPI_Recv(&signal, 1, MPI_INT, 0, tag_signal, globComm, MPI_STATUS_IGNORE);
                    if (signal == -1) {
                        simulation_running = false;
                    }
                }
                if (simulation_running) {
                    MPI_Send(global_vegetation.data(), global_vegetation.size(), MPI_UINT8_T, 0, tag_veg, globComm);
                    MPI_Send(global_fire.data(), global_fire.size(), MPI_UINT8_T, 0, tag_fire, globComm);
                }
            }
        }

        // Envoi du signal de fin au processus d'affichage
        if (comp_rank == 0) {
            int signal = -1;
            MPI_Send(&signal, 1, MPI_INT, 0, tag_signal, globComm);
        }
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
