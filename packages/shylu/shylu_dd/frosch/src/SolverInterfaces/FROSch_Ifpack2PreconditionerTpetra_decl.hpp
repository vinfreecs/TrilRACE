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

#ifndef _FROSCH_IFPACK2PRECONDITIONERTPETRA_DECL_HPP
#define _FROSCH_IFPACK2PRECONDITIONERTPETRA_DECL_HPP

#include <ShyLU_DDFROSch_config.h>

#include <Ifpack2_ReorderFilter_decl.hpp>
#include <Ifpack2_Details_OneLevelFactory_decl.hpp>
#include <Tpetra_BlockCrsMatrix_decl.hpp> // VR 2026-11-03

// FROSch
#include <FROSch_Solver_def.hpp>


namespace FROSch {

    using namespace Teuchos;
    using namespace Xpetra;

    template <class SC = double,
              class LO = int,
              class GO = DefaultGlobalOrdinal,
              class NO = KokkosClassic::DefaultNode::DefaultNodeType>
    class Ifpack2PreconditionerTpetra : public Solver<SC,LO,GO,NO> {

    protected:

        // Xpetra
        using ConstXMatrixPtr                   = typename Solver<SC,LO,GO,NO>::ConstXMatrixPtr;

        using XMultiVector                      = typename Solver<SC,LO,GO,NO>::XMultiVector;
        using XMultiVectorPtr                   = typename Solver<SC,LO,GO,NO>::XMultiVectorPtr;

        using XMultiVectorFactory               = typename Solver<SC,LO,GO,NO>::XMultiVectorFactory;

        // Tpetra
        using TOperator                         = Tpetra::Operator<SC,LO,GO,NO>;

        using TRowMatrix                        = Tpetra::RowMatrix<SC,LO,GO,NO>;
        using TRowMatrixPtr                     = RCP<TRowMatrix>;
        using ConstTRowMatrixPtr                = RCP<const TRowMatrix>;

        using TCrsMatrix                        = Tpetra::CrsMatrix<SC,LO,GO,NO>;
        using TCrsMatrixPtr                     = RCP<TCrsMatrix>;
        using ConstTCrsMatrixPtr                = RCP<const TCrsMatrix>;

        using TMultiVector                      = Tpetra::MultiVector<SC,LO,GO,NO>;
        using TMultiVectorPtr                   = RCP<TMultiVector>;

        // Teuchos
        using ParameterListPtr                  = typename Solver<SC,LO,GO,NO>::ParameterListPtr;

        // Ifpack2
        using Ifpack2PreconditionerPtr          = RCP<Ifpack2::Preconditioner<SC,LO,GO,NO> >;
        using TRowMatrixFilterType              = Ifpack2::ReorderFilter<TRowMatrix>;

        // VR 2026-11-03: "Local Matrix Format" = "BlockCrs"
        using TMap                              = Tpetra::Map<LO,GO,NO>;
        using TCrsGraph                         = Tpetra::CrsGraph<LO,GO,NO>;
        using TBlockCrsMatrix                   = Tpetra::BlockCrsMatrix<SC,LO,GO,NO>;
        using TVector                           = Tpetra::Vector<SC,LO,GO,NO>;
        using HostLOView                        = Kokkos::View<LO*,Kokkos::HostSpace>;

    public:

        //! Initialize the internal solver
        virtual int initialize();

        //! Compute the internal solver
        virtual int compute();

        /*!
        \brief Computes the operator-multivector application.

        Loosely, performs \f$Y = \alpha \cdot A^{\textrm{mode}} \cdot X + \beta \cdot Y\f$. However, the details of operation
        vary according to the values of \c alpha and \c beta. Specifically
        - if <tt>beta == 0</tt>, apply() <b>must</b> overwrite \c Y, so that any values in \c Y (including NaNs) are ignored.
        - if <tt>alpha == 0</tt>, apply() <b>may</b> short-circuit the operator, so that any values in \c X (including NaNs) are ignored.
        */
        void apply(const XMultiVector &x,
                   XMultiVector &y,
                   ETransp mode=NO_TRANS,
                   SC alpha=ScalarTraits<SC>::one(),
                   SC beta=ScalarTraits<SC>::zero()) const;

        int updateMatrix(ConstXMatrixPtr k,
                        bool reuseInitialize=false);

    protected:

        //! Constructor
        Ifpack2PreconditionerTpetra(ConstXMatrixPtr k,
                                    ParameterListPtr parameterList,
                                    string description);

        Ifpack2PreconditionerPtr Ifpack2Preconditioner_ = null;

        bool useRILUK;
        bool useZoltan2;
        bool needToApplyPerm;
        Teuchos::ArrayRCP<LO>    perm;
        Teuchos::ArrayRCP<LO>    revperm;
        int globalRank_ = 0;

        // VR 2026-11-03: local matrix as a Tpetra::BlockCrsMatrix (Chebyshev only), nodes found from the pattern
        void buildBlockCrsStructure();
        void fillBlockCrsValues();
        void checkBlockCrsApply() const;

        bool useBlockCrs_ = false;
        LO blockSize_ = 3;
        ConstTCrsMatrixPtr blockPointMatrix_ = null;    // the point matrix FROSch keeps updating in place
        RCP<TCrsGraph> blockGraph_ = null;
        RCP<TBlockCrsMatrix> blockMatrix_ = null;
        RCP<TVector> blockInvDiagonal_ = null;          // given to Chebyshev ("chebyshev: operator inv diagonal")
        ParameterList blockIfpack2Params_;              // Ifpack2 list; set again in compute(): Ifpack2 deep-copies the diagonal in setParameters()
        SC blockMinDiagonal_ = ScalarTraits<SC>::eps();  // as Ifpack2 Chebyshev's "chebyshev: min diagonal value"
        HostLOView blockPerm_;                          // point row: old -> new
        HostLOView blockInvPerm_;                       // point row: new -> old
        HostLOView blockColToRow_;                      // local column index of the point matrix -> local row index
        bool blockChecked_ = false;
        bool blockIdentity_ = false;                    // permutation = identity (input already node-ordered): no gather / scatter
        mutable RCP<TMultiVector> blockX_ = null;
        mutable RCP<TMultiVector> blockY_ = null;

        friend class SolverFactory<SC,LO,GO,NO>;
    };

}

#endif
