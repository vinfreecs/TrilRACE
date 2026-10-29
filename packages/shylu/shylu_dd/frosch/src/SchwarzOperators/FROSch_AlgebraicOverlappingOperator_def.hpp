//@HEADER
// ************************************************************************
//
//               ShyLU: Hybrid preconditioner package
//                 Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Alexander Heinlein (alexander.heinlein@uni-koeln.de)
//
// ************************************************************************
//@HEADER

#ifndef _FROSCH_ALGEBRAICOVERLAPPINGOPERATOR_DEF_HPP
#define _FROSCH_ALGEBRAICOVERLAPPINGOPERATOR_DEF_HPP

#include <FROSch_AlgebraicOverlappingOperator_decl.hpp>

// VR 2026.10.29: overlap dumps (FROSCH_DUMP_DIR)
#include <cstdlib>
#include <fstream>
#include <Xpetra_IO.hpp>


namespace FROSch {

    using namespace std;
    using namespace Teuchos;
    using namespace Xpetra;

    template <class SC,class LO,class GO,class NO>
    AlgebraicOverlappingOperator<SC,LO,GO,NO>::AlgebraicOverlappingOperator(ConstXMatrixPtr k,
                                                                            ParameterListPtr parameterList) :
    OverlappingOperator<SC,LO,GO,NO> (k,parameterList)
    {
        FROSCH_DETAILTIMER_START_LEVELID(algebraicOverlappingOperatorTime,"AlgebraicOverlappingOperator::AlgebraicOverlappingOperator");
        if (!this->ParameterList_->get("Adding Layers Strategy","CrsGraph").compare("CrsGraph")) {
            AddingLayersStrategy_ = LayersFromGraph;
        } else if (!this->ParameterList_->get("Adding Layers Strategy","CrsGraph").compare("CrsMatrix")) {
            AddingLayersStrategy_ = LayersFromMatrix;
        } else if (!this->ParameterList_->get("Adding Layers Strategy","CrsGraph").compare("Old")) {
            AddingLayersStrategy_ = LayersOld;
        } else {
            FROSCH_ASSERT(false,"FROSch::AlgebraicOverlappingOperator: Specify a valid strategy for adding layers.");
        }
        // VR 2026.10.29: overlap dumps for the before/after checks
        if (const char* dumpDir = std::getenv("FROSCH_DUMP_DIR")) DumpDir_ = dumpDir;
    }

    template <class SC,class LO,class GO,class NO>
    int AlgebraicOverlappingOperator<SC,LO,GO,NO>::initialize(int overlap,
                                                              ConstXMapPtr repeatedMap)
    {
        FROSCH_TIMER_START_LEVELID(initializeTime,"AlgebraicOverlappingOperator::initialize");

        if (this->Verbose_) {
            cout
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << setw(89) << "-----------------------------------------------------------------------------------------"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| "
            << left << setw(74) << "AlgebraicOverlappingOperator " << right << setw(8) << "(Level " << setw(2) << this->LevelID_ << ")"
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << setw(89) << "========================================================================================="
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(41) << "Adding layers strategy" << right
            << " | " << setw(41) << this->ParameterList_->get("Adding Layers Strategy","CrsGraph")
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(41) << "Combine mode in overlap" << right
            << " | " << setw(41) << this->ParameterList_->get("Combine Values in Overlap","Restricted")
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(41) << "Solver type" << right
            << " | " << setw(41) << this->ParameterList_->sublist("Solver").get("SolverType","Amesos2")
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(41) << "Solver" << right
            << " | " << setw(41) << this->ParameterList_->sublist("Solver").get("Solver","Klu")
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(41) << "Reuse symbolic factorization" << right
            << " | " << setw(41) << this->ParameterList_->get("Reuse: Symbolic Factorization",true)
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << setw(89) << "-----------------------------------------------------------------------------------------"
            << endl;
        }

        if (repeatedMap.is_null()) repeatedMap = this->K_->getRangeMap();
        this->buildOverlappingMatrices(overlap,repeatedMap);
        this->initializeOverlappingOperator();
        this->updateLocalOverlappingMatrices_Symbolic();
        // VR 2026.10.26: rows and nnz of the overlapped local subdomain matrix (what the subdomain solver gets),
        // to match the overlap of different rank/thread layouts by data per core.
        if (!this->ParameterList_->get("Verbosity","All").compare("All") && !this->localSubdomainMatrix_.is_null()) {
            GO localRows = (GO) this->localSubdomainMatrix_->getLocalNumRows();
            GO localNnz = (GO) this->localSubdomainMatrix_->getLocalNumEntries();
            GO sumRows = 0, minRows = 0, maxRows = 0, sumNnz = 0, minNnz = 0, maxNnz = 0;
            reduceAll(*this->MpiComm_,REDUCE_SUM,localRows,ptr(&sumRows));
            reduceAll(*this->MpiComm_,REDUCE_MIN,localRows,ptr(&minRows));
            reduceAll(*this->MpiComm_,REDUCE_MAX,localRows,ptr(&maxRows));
            reduceAll(*this->MpiComm_,REDUCE_SUM,localNnz,ptr(&sumNnz));
            reduceAll(*this->MpiComm_,REDUCE_MIN,localNnz,ptr(&minNnz));
            reduceAll(*this->MpiComm_,REDUCE_MAX,localNnz,ptr(&maxNnz));
            double numRanks = double(this->MpiComm_->getSize());
            if (this->Verbose_) {
                cout
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << setw(89) << "-----------------------------------------------------------------------------------------"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << "| "
                << left << setw(74) << "> Local Subdomain Matrix Statistics " << right << setw(8) << "(Level " << setw(2) << this->LevelID_ << ")"
                << " |"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << setw(89) << "========================================================================================="
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << "| " << left << setw(20) << " " << right
                << " | " << setw(10) << "ranks"
                << " | " << setw(10) << "avg"
                << " | " << setw(10) << "min"
                << " | " << setw(10) << "max"
                << " | " << setw(10) << "global sum"
                << " |"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << setw(89) << "-----------------------------------------------------------------------------------------"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << "| " << left << setw(20) << "Local rows" << right
                << " | " << setw(10) << this->MpiComm_->getSize()
                << " | " << setw(10) << setprecision(8) << sumRows/numRanks
                << " | " << setw(10) << minRows
                << " | " << setw(10) << maxRows
                << " | " << setw(10) << sumRows
                << " |"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << "| " << left << setw(20) << "Local nnz" << right
                << " | " << setw(10) << this->MpiComm_->getSize()
                << " | " << setw(10) << setprecision(10) << sumNnz/numRanks
                << " | " << setw(10) << minNnz
                << " | " << setw(10) << maxNnz
                << " | " << setw(10) << sumNnz
                << " |"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << setw(89) << "-----------------------------------------------------------------------------------------"
                << endl;
            }
        }
        bool reuseSymbolicFactorization = this->ParameterList_->get("Reuse: Symbolic Factorization",true);
        if (this->ExtractLocalSubdomainMatrix_Symbolic_Done_ && reuseSymbolicFactorization) {
            // if reuseSymbolicFactorization=false, we call initializeSubdomainSolver is called during compute
            // Inject global MPI rank into Solver parameters so subdomain solver can use it for diagnostics
            this->ParameterList_->sublist("Solver").set("Global MPI Rank", this->MpiComm_->getRank());
            this->initializeSubdomainSolver(this->localSubdomainMatrix_);
        }

        this->IsInitialized_ = true;
        this->IsComputed_ = false;
        return 0; // RETURN VALUE!!!
    }

    template <class SC,class LO,class GO,class NO>
    int AlgebraicOverlappingOperator<SC,LO,GO,NO>::compute()
    {
        FROSCH_TIMER_START_LEVELID(computeTime,"AlgebraicOverlappingOperator::compute");
        FROSCH_ASSERT(this->IsInitialized_,"ERROR: AlgebraicOverlappingOperator has to be initialized before calling compute()");
        this->computeOverlappingOperator();
        // VR 2026.10.29: D3, the local subdomain matrix the solver gets (18 significant digits, no exit)
        if (!DumpDir_.empty()) {
            Xpetra::IO<SC,LO,GO,NO>::Write(DumpDir_ + "/local_rank" + to_string(this->MpiComm_->getRank()) + ".mtx",
                                           *this->OverlappingMatrix_);
        }
        // DL 2026.04.09: For easy benchmarking of local subdomain problems!
#ifdef PRINT_LOCAL_MTX
        int myRank = this->MpiComm_->getRank();
        std::string filename = "local_subdomain_rank_" + std::to_string(myRank) + ".mtx";
        std::cout << "Rank: " << myRank << " writing its local subdomain system to: " << filename << std::endl;
        this->exportLocalMatrix(filename);
        std::cout << "Rank: " << myRank << " finished writing" << std::endl;
        std::fflush(stdout); // ensure output is flushed before exit
        std::exit(EXIT_SUCCESS);
#endif
        return 0; // RETURN VALUE!!!
    }

    template <class SC,class LO,class GO,class NO>
    void AlgebraicOverlappingOperator<SC,LO,GO,NO>::describe(FancyOStream &out,
                                                             const EVerbosityLevel verbLevel) const
    {
        FROSCH_ASSERT(false,"describe() has to be implemented properly...");
    }

    template <class SC,class LO,class GO,class NO>
    string AlgebraicOverlappingOperator<SC,LO,GO,NO>::description() const
    {
        return "Algebraic Overlapping Operator";
    }

    // VR 2026.10.29: one GID per line, in map order
    template <class SC,class LO,class GO,class NO>
    void AlgebraicOverlappingOperator<SC,LO,GO,NO>::dumpMapGIDs(ConstXMapPtr map,
                                                                const string& name) const
    {
        ofstream out(DumpDir_ + "/" + name + "_rank" + to_string(this->MpiComm_->getRank()) + ".txt");
        ArrayView<const GO> gids = map->getLocalElementList();
        for (LO i=0; i<(LO) gids.size(); i++) out << gids[i] << "\n";
    }

    // VR 2026.10.29: one line per owned row: row GID, then its column GIDs in stored order
    template <class SC,class LO,class GO,class NO>
    void AlgebraicOverlappingOperator<SC,LO,GO,NO>::dumpGraphRows(ConstXCrsGraphPtr graph,
                                                                  const string& name) const
    {
        ofstream out(DumpDir_ + "/" + name + "_rank" + to_string(this->MpiComm_->getRank()) + ".txt");
        for (LO i=0; i<(LO) graph->getLocalNumRows(); i++) {
            ArrayView<const LO> indices;
            graph->getLocalRowView(i,indices);
            out << graph->getRowMap()->getGlobalElement(i);
            for (LO j=0; j<(LO) indices.size(); j++) out << " " << graph->getColMap()->getGlobalElement(indices[j]);
            out << "\n";
        }
    }

    template <class SC,class LO,class GO,class NO>
    int AlgebraicOverlappingOperator<SC,LO,GO,NO>::buildOverlappingMatrices(int overlap,
                                                                            ConstXMapPtr repeatedMap)
    {
        FROSCH_DETAILTIMER_START_LEVELID(buildOverlappingMatricesTime,"AlgebraicOverlappingOperator::buildOverlappingMatrices");
        // ====================================================================================
        // AH 08/09/2019: This is just temporary. Implement this properly in all the classes
        Verbosity verbosity = All;
        if (!this->ParameterList_->get("Verbosity","All").compare("None")) {
            verbosity = None;
        } else if (!this->ParameterList_->get("Verbosity","All").compare("All")) {
            verbosity = All;
        } else {
            FROSCH_ASSERT(false,"FROSch::AlgebraicOverlappingOperator: Specify a valid verbosity level.");
        }
        // ====================================================================================

        this->OverlappingMap_ = repeatedMap;
        this->OverlappingMatrix_ = this->K_;

        // VR 2026.10.29: D0 = K's owned rows, D1 layer 0 = the starting map
        if (!DumpDir_.empty()) {
            dumpGraphRows(this->K_->getCrsGraph(),"kgraph");
            dumpMapGIDs(this->OverlappingMap_,"map_layer0");
        }

        GO global = 0, sum = 0;
        LO local,minVal,maxVal;
        SC avg;
        if (verbosity==All) {
            FROSCH_DETAILTIMER_START_LEVELID(printStatisticsTime,"print statistics");

            global = this->OverlappingMap_->getMaxAllGlobalIndex();
            if (this->OverlappingMap_->lib()==UseEpetra || this->OverlappingMap_->getGlobalNumElements()>0) {
                global += 1;
            }

            local = (LO) max((LO) this->OverlappingMap_->getLocalNumElements(),(LO) 0);
            reduceAll(*this->MpiComm_,REDUCE_SUM,GO(local),ptr(&sum));
            avg = max(sum/double(this->MpiComm_->getSize()),0.0);
            reduceAll(*this->MpiComm_,REDUCE_MIN,local,ptr(&minVal));
            reduceAll(*this->MpiComm_,REDUCE_MAX,local,ptr(&maxVal));

            if (this->Verbose_) {
                cout
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << setw(89) << "-----------------------------------------------------------------------------------------"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << "| "
                << left << setw(74) << "> Overlapping Subdomains Statistics " << right << setw(8) << "(Level " << setw(2) << this->LevelID_ << ")"
                << " |"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << setw(89) << "========================================================================================="
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << "| " << left << setw(20) << " " << right
                << " | " << setw(10) << "total"
                << " | " << setw(10) << "avg"
                << " | " << setw(10) << "min"
                << " | " << setw(10) << "max"
                << " | " << setw(10) << "global sum"
                << " |"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << setw(89) << "-----------------------------------------------------------------------------------------"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << "| " << left << setw(20) << "Layer 0" << right
                << " | " << setw(10) << global
                << " | " << setw(10) << setprecision(5) << avg
                << " | " << setw(10) << minVal
                << " | " << setw(10) << maxVal
                << " | " << setw(10) << sum
                << " |";
            }
        }

        // Adding Layers of Elements to the overlapping subdomains
        ConstXCrsGraphPtr overlappingGraph = this->OverlappingMatrix_->getCrsGraph();
        for (int i=0; i<overlap; i++) {
            switch (AddingLayersStrategy_) {
                case LayersFromGraph:
                    ExtendOverlapByOneLayer(overlappingGraph,this->OverlappingMap_,overlappingGraph,this->OverlappingMap_);
                    break;

                case LayersFromMatrix:
                    ExtendOverlapByOneLayer(this->OverlappingMatrix_,this->OverlappingMap_,this->OverlappingMatrix_,this->OverlappingMap_);
                    break;

                case LayersOld:
                    ExtendOverlapByOneLayer_Old(this->OverlappingMatrix_,this->OverlappingMap_,this->OverlappingMatrix_,this->OverlappingMap_);
                    break;

                default:
                    FROSCH_ASSERT(false,"FROSch::AlgebraicOverlappingOperator: Specify a valid strategy for adding layers.");
            }
            // VR 2026.10.29: D1 = the map after layer i+1
            if (!DumpDir_.empty()) dumpMapGIDs(this->OverlappingMap_,"map_layer"+to_string(i+1));
            if (verbosity==All) {
                FROSCH_DETAILTIMER_START_LEVELID(printStatisticsTime,"print statistics");
                local = (LO) max((LO) this->OverlappingMap_->getLocalNumElements(),(LO) 0);
                reduceAll(*this->MpiComm_,REDUCE_SUM,GO(local),ptr(&sum));
                avg = max(sum/double(this->MpiComm_->getSize()),0.0);
                reduceAll(*this->MpiComm_,REDUCE_MIN,local,ptr(&minVal));
                reduceAll(*this->MpiComm_,REDUCE_MAX,local,ptr(&maxVal));

                if (this->Verbose_) {
                    cout
                    << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                    << "| " << left << "Layer " << setw(14) << i+1 << right
                    << " | " << setw(10) << global
                    << " | " << setw(10) << setprecision(5) << avg
                    << " | " << setw(10) << minVal
                    << " | " << setw(10) << maxVal
                    << " | " << setw(10) << sum
                    << " |";
                }
            }
        }

        if (verbosity==All) {
            FROSCH_DETAILTIMER_START_LEVELID(printStatisticsTime,"print statistics");
            if (this->Verbose_) {
                cout
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << setw(89) << "-----------------------------------------------------------------------------------------"
                << endl;
            }
        }

        // AH 08/28/2019 TODO: It seems that ExtendOverlapByOneLayer_Old is currently the fastest method because the map is sorted. This seems to be better for the direct solver. (At least Klu)
        if (this->ParameterList_->get("Sort Overlapping Map",true)) {
            switch (AddingLayersStrategy_) {
                case LayersFromGraph:
                    this->OverlappingMap_ = SortMapByGlobalIndex(this->OverlappingMap_);
                    break;

                case LayersFromMatrix:
                    this->OverlappingMap_ = SortMapByGlobalIndex(this->OverlappingMap_);
                    break;

                case LayersOld:
                    if (this->Verbose_) cout << "FROSch::AlgebraicOverlappingOperator : The overlapping map is already sorted" << endl;
                    break;

                default:
                    FROSCH_ASSERT(false,"FROSch::AlgebraicOverlappingOperator: Specify a valid strategy for adding layers.");
            }
        }

        // VR 2026.10.29: D2 = the final (sorted) overlap map
        if (!DumpDir_.empty()) dumpMapGIDs(this->OverlappingMap_,"map_final");

        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int AlgebraicOverlappingOperator<SC,LO,GO,NO>::updateLocalOverlappingMatrices()
    {
        FROSCH_DETAILTIMER_START_LEVELID(updateLocalOverlappingMatricesTime,"AlgebraicOverlappingOperator::updateLocalOverlappingMatrices");
        if (this->ExtractLocalSubdomainMatrix_Symbolic_Done_) {
            // using original K_ as input
            ExtractLocalSubdomainMatrix_Compute(this->K_, this->subdomainMatrix_, this->localSubdomainMatrix_);
            this->OverlappingMatrix_ = this->localSubdomainMatrix_.getConst();
        } else {
            if (this->IsComputed_) {
                // already computed once and we want to recycle the information. That is why we reset OverlappingMatrix_ to K_, because K_ has been reset at this point
                this->OverlappingMatrix_ = this->K_;
            }
            this->OverlappingMatrix_ = ExtractLocalSubdomainMatrix(this->OverlappingMatrix_, this->OverlappingMap_);
        }
        return 0;
    }


    template <class SC,class LO,class GO,class NO>
    int AlgebraicOverlappingOperator<SC,LO,GO,NO>::updateLocalOverlappingMatrices_Symbolic()
    {
        FROSCH_DETAILTIMER_START_LEVELID(updateLocalOverlappingMatrices_SymbolicTime, "AlgebraicOverlappingOperator::updateLocalOverlappingMatrices_Symbolic");
        this->extractLocalSubdomainMatrix_Symbolic();
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    void AlgebraicOverlappingOperator<SC,LO,GO,NO>::extractLocalSubdomainMatrix_Symbolic()
    {
        if (this->OverlappingMap_->lib() == UseTpetra) {
            FROSCH_DETAILTIMER_START_LEVELID(AlgebraicOverlappin_extractLocalSubdomainMatrix_SymbolicTime,"AlgebraicOverlappinOperator::extractLocalSubdomainMatrix_Symbolic");
            // buid sudomain matrix
            this->subdomainMatrix_ = MatrixFactory<SC,LO,GO,NO>::Build(this->OverlappingMap_, this->OverlappingMatrix_->getGlobalMaxNumRowEntries());
            RCP<Import<LO,GO,NO> > scatter = ImportFactory<LO,GO,NO>::Build(this->OverlappingMatrix_->getRowMap(), this->OverlappingMap_);
            this->subdomainMatrix_->doImport(*(this->OverlappingMatrix_), *scatter, ADD);

            // build local subdomain matrix
            RCP<const Comm<LO> > SerialComm = rcp(new MpiComm<LO>(MPI_COMM_SELF));
            RCP<Map<LO,GO,NO> > localSubdomainMap = MapFactory<LO,GO,NO>::Build(this->OverlappingMap_->lib(), this->OverlappingMap_->getLocalNumElements(), 0, SerialComm);
            this->localSubdomainMatrix_ = MatrixFactory<SC,LO,GO,NO>::Build(localSubdomainMap, localSubdomainMap, this->OverlappingMatrix_->getGlobalMaxNumRowEntries());

            // fill in column indexes
            ExtractLocalSubdomainMatrix_Symbolic(this->subdomainMatrix_, // input
                                                 this->localSubdomainMatrix_);   // output

            // turn flag on
            this->ExtractLocalSubdomainMatrix_Symbolic_Done_ = true;
        }
    }
}

#endif
