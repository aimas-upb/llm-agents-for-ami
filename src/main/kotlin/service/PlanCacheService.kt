package org.eclipse.lmos.arc.app.service

import org.springframework.stereotype.Service

@Service
class PlanCacheService {

    private var latestPlanJson: String? = null
    private val lock = Any() // For basic thread safety

    fun storePlan(planJson: String) {
        synchronized(lock) {
            latestPlanJson = planJson
        }
    }

    fun retrieveAndClearPlan(): String? {
        synchronized(lock) {
            val planToReturn = latestPlanJson
            latestPlanJson = null
            return planToReturn
        }
    }
} 