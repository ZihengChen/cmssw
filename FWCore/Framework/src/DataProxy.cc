// -*- C++ -*-
//
// Package:     Framework
// Class  :     DataProxy
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Thu Mar 31 12:49:19 EST 2005
//

// system include files
#include <mutex>

// user include files
#include "FWCore/Framework/interface/DataProxy.h"
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Framework/interface/MakeDataException.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/src/ESGlobalMutex.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Concurrency/interface/WaitingTaskList.h"

#include "tbb/task_arena.h"
namespace edm {
  namespace eventsetup {

    static const ComponentDescription* dummyDescription() {
      static ComponentDescription s_desc;
      return &s_desc;
    }

    DataProxy::DataProxy()
        : description_(dummyDescription()),
          cache_(nullptr),
          cacheIsValid_(false),
          nonTransientAccessRequested_(false) {}

    DataProxy::~DataProxy() {}

    void DataProxy::clearCacheIsValid() {
      nonTransientAccessRequested_.store(false, std::memory_order_release);
      cache_ = nullptr;
      cacheIsValid_.store(false, std::memory_order_release);
    }

    void DataProxy::resetIfTransient() {
      if (!nonTransientAccessRequested_.load(std::memory_order_acquire)) {
        clearCacheIsValid();
        invalidateTransientCache();
      }
    }

    void DataProxy::invalidateTransientCache() { invalidateCache(); }

    namespace {
      void throwMakeException(const EventSetupRecordImpl& iRecord, const DataKey& iKey) {
        throw MakeDataException(iRecord.key(), iKey);
      }

      class ESSignalSentry {
      public:
        ESSignalSentry(const EventSetupRecordImpl& iRecord,
                       const DataKey& iKey,
                       ComponentDescription const* componentDescription,
                       ActivityRegistry const* activityRegistry)
            : eventSetupRecord_(iRecord),
              dataKey_(iKey),
              componentDescription_(componentDescription),
              calledPostLock_(false),
              activityRegistry_(activityRegistry) {
          activityRegistry->preLockEventSetupGetSignal_(componentDescription_, eventSetupRecord_.key(), dataKey_);
        }
        void sendPostLockSignal() {
          calledPostLock_ = true;
          activityRegistry_->postLockEventSetupGetSignal_(componentDescription_, eventSetupRecord_.key(), dataKey_);
        }
        ~ESSignalSentry() noexcept(false) {
          if (!calledPostLock_) {
            activityRegistry_->postLockEventSetupGetSignal_(componentDescription_, eventSetupRecord_.key(), dataKey_);
          }
          activityRegistry_->postEventSetupGetSignal_(componentDescription_, eventSetupRecord_.key(), dataKey_);
        }

      private:
        EventSetupRecordImpl const& eventSetupRecord_;
        DataKey const& dataKey_;
        ComponentDescription const* componentDescription_;
        bool calledPostLock_;
        ActivityRegistry const* activityRegistry_;
      };
    }  // namespace

    void DataProxy::prefetchAsync(WaitingTask* iTask,
                                  EventSetupRecordImpl const& iRecord,
                                  DataKey const& iKey,
                                  EventSetupImpl const* iEventSetupImpl) const {
      const_cast<DataProxy*>(this)->prefetchAsyncImpl(iTask, iRecord, iKey, iEventSetupImpl);
    }

    void const* DataProxy::getAfterPrefetch(const EventSetupRecordImpl& iRecord,
                                            const DataKey& iKey,
                                            bool iTransiently) const {
      //We need to set the AccessType for each request so this can't be called in an earlier function in the stack.
      //This also must be before the cache_ check since we want to setCacheIsValid before a possible
      // exception throw. If we don't, 'getImpl' will be called again on a second request for the data.

      if
        LIKELY(!iTransiently) { nonTransientAccessRequested_.store(true, std::memory_order_release); }

      if
        UNLIKELY(!cacheIsValid()) {
          cache_ = getAfterPrefetchImpl();
          cacheIsValid_.store(true, std::memory_order_release);
        }

      if
        UNLIKELY(nullptr == cache_) { throwMakeException(iRecord, iKey); }
      return cache_;
    }

    const void* DataProxy::get(const EventSetupRecordImpl& iRecord,
                               const DataKey& iKey,
                               bool iTransiently,
                               ActivityRegistry const* activityRegistry,
                               EventSetupImpl const* iEventSetupImpl) const {
      if (!cacheIsValid()) {
        ESSignalSentry signalSentry(iRecord, iKey, providerDescription(), activityRegistry);
        std::lock_guard<std::recursive_mutex> guard(esGlobalMutex());
        signalSentry.sendPostLockSignal();
        if (!cacheIsValid()) {
          auto waitTask = edm::make_empty_waiting_task();
          waitTask->set_ref_count(2);
          auto waitTaskPtr = waitTask.get();
          tbb::this_task_arena::isolate([this, waitTaskPtr, &iRecord, &iKey, iEventSetupImpl]() {
            prefetchAsync(waitTaskPtr, iRecord, iKey, iEventSetupImpl);
            waitTaskPtr->decrement_ref_count();
            waitTaskPtr->wait_for_all();
          });
          cache_ = getAfterPrefetchImpl();
          cacheIsValid_.store(true, std::memory_order_release);
          if (waitTask->exceptionPtr()) {
            std::rethrow_exception(*waitTask->exceptionPtr());
          }
        }
      }
      return getAfterPrefetch(iRecord, iKey, iTransiently);
    }

    void DataProxy::doGet(const EventSetupRecordImpl& iRecord,
                          const DataKey& iKey,
                          bool iTransiently,
                          ActivityRegistry const* activityRegistry,
                          EventSetupImpl const* iEventSetupImpl) const {
      get(iRecord, iKey, iTransiently, activityRegistry, iEventSetupImpl);
    }

  }  // namespace eventsetup
}  // namespace edm
