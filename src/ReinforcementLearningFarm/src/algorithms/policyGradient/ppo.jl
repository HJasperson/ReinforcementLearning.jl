export PPOPolicy

using Distributions
using Flux
using Random

"""
    PPOPolicy(;kwargs)

# Keyword arguments

- `learner`,
- `explorer`,
- `device`,
- `γ = 0.99f0`,
- `λ = 0.95f0`,
- `clip_range = 0.2f0`,
- `max_grad_norm = 0.5f0`,
- `minibatches = 4`,
- `n_epochs = 4`,
- `actor_loss_weight = 1.0f0`,
- `critic_loss_weight = 0.5f0`,
- `entropy_loss_weight = 0.01f0`,
- `dist = Categorical`,
- `rng = Random.GLOBAL_RNG`,

By default, `dist` is set to `Categorical`, which means it will only works
on environments of discrete actions. To work with environments of continuous
actions `dist` should be set to `Normal` and the `actor` in the `learner`
should be a `GaussianNetwork`. Using it with a `GaussianNetwork` supports 
multi-dimensional action spaces, though it only supports it under the assumption
that the dimensions are independent since the `GaussianNetwork` outputs a single
`μ` and `σ` for each dimension which is used to simplify the calculations.

n_epochs and minibatches are for the outer and inner gradient update loops, respectively.
device can be a cpu or gpu.
"""
mutable struct PPOPolicy{A<:ActorCritic,E<:AbstractExplorer,F<:Flux.AbstractDevice,
    D<:Distributions.ValueSupport,R} <: AbstractPolicy
    learner::A
    explorer::E
    device::F
    dist::D
    γ::Float32
    λ::Float32
    clip_range::Float32
    max_grad_norm::Float32
    minibatches::Int
    n_epochs::Int
    actor_loss_weight::Float32
    critic_loss_weight::Float32
    entropy_loss_weight::Float32
    rng::R
end

function PPOPolicy(;
    learner,
    explorer,
    dist = Distributions.Discrete,
    γ = 0.99f0,
    λ = 0.95f0,
    clip_range = 0.2f0,
    max_grad_norm = 0.5f0,
    minibatches = 4,
    n_epochs = 4,
    actor_loss_weight = 1.0f0,
    critic_loss_weight = 0.5f0,
    entropy_loss_weight = 0.01f0,
    rng = Random.GLOBAL_RNG,
    device = Flux.get_device(learner)
)
    PPOPolicy{typeof(learner),typeof(explorer),typeof(device),dist,typeof(rng)}(
        learner,
        explorer,
        dist,
        γ,
        λ,
        clip_range,
        max_grad_norm,
        minibatches,
        n_epochs,
        actor_loss_weight,
        critic_loss_weight,
        entropy_loss_weight,
        rng,
        device,
    )
end

Flux.@layer PPOPolicy trainable=(learner,)

function RLBase.plan!(p::PPOPolicy, env::E) where {E<:AbstractEnv}
    RLBase.plan!(p, env, p.dist)
end

function RLBase.plan!(p::PPOPolicy, env::E, ::Distributions.Continuous) where {E<:AbstractEnv}
    # run the actor
    μ, logσ = p.learner.actor(env |> p.device) |> cpu

    # run the explorer

end

function RLBase.plan!(p::PPOPolicy, env::E, ::Distributions.Discrete) where {E<:AbstractEnv}
    # run the actor to get prob distribution
    dist = RLBase.prob(p, env, p.dist)
    logprobs = log.(probs(dist))

    # run the explorer
    action = RLBase.plan!(p.explorer, logprobs)
end


function RLBase.prob(p::PPOPolicy, env::E) where {E<:AbstractEnv}
    RLBase.prob(p, env, p.dist)
end

function RLBase.prob(p::PPOPolicy, env::E, ::Distributions.Continuous) where {E<:AbstractEnv}
    
end

function RLBase.prob(p::PPOPolicy, env::E, ::Distributions.Discrete) where {E<:AbstractEnv}
    logits = p.learner.actor(env |> p.device)
    # todo: masking
    probs = softmax(logits) |> cpu
    return Categorical(probs)
end


function RLBase.optimize!(p::PPOPolicy, ::PostActStage, traj::Trajectory)
    # check if it's time to optimize via the controller
    if on_sample!(traj.controller)
        batch = sample(traj.sampler, traj.container)    # run the sampler
        batch_size = length(batch[:terminal])  # todo: check if this is correct
        minibatch_size = batch_size / p.minibatches

        states_plus = batch[:state] |> p.device
        states_flatten = flatten_batch(select_last_dim(states_plus, 1:batch_size))
        values = p.learner.critic(flatten_batch(states_plus)) |> cpu    # todo: might need to be reshaped
        advantages = generalized_advantage_estimation(
            batch[:reward],
            values,
            p.γ,
            p.λ;
            dims = 2,
            terminal = batch[:terminal],
        )
        returns = advantages .+ values  # todo: not sure this is the correct VF loss

        actions_flatten = flatten_batch(select_last_dim(batch[:action], 1:batch_size))
        action_log_probs = select_last_dim(batch[:action_log_prob], 1:batch_size)

    end
end



#########################################################################################

function RLBase.prob(
    p::PPOPolicy{<:ActorCritic{<:GaussianNetwork},Normal},
    state::AbstractArray,
)
    if p.update_step < p.n_random_start
        @error "todo"
    else
        μ, logσ = p.learner.actor(state |> p.device) |> cpu
        StructArray{Normal}((μ, exp.(logσ)))
    end
end

function RLBase.prob(p::PPOPolicy{<:ActorCritic,Categorical}, state::AbstractArray)
    state |> p.device
    logprobs = p.learner.actor(state |> p.device) |> softmax |> cpu
    if p.update_step < p.n_random_start
        [
            Categorical(fill(1 / length(x), length(x)); check_args = false) for
            x in eachcol(logprobs)
        ]
    else
        [Categorical(x; check_args = false) for x in eachcol(logprobs)]
    end
end

RLBase.prob(p::PPOPolicy, env::MultiThreadEnv) = prob(p, state(env))

function RLBase.prob(p::PPOPolicy, env::AbstractEnv)
    s = state(env)
    s = Flux.unsqueeze(s, ndims(s) + 1)
    prob(p, s)
end

(p::PPOPolicy)(env::MultiThreadEnv) = rand.(p.rng, prob(p, env))
(p::PPOPolicy)(env::AbstractEnv) = rand.(p.rng, prob(p, env))

function (agent::Agent{<:PPOPolicy})(env::MultiThreadEnv)
    dist = prob(agent.policy, env)
    action = rand.(agent.policy.rng, dist)
    if ndims(action) == 2
        action_log_prob = sum(logpdf.(dist, action), dims = 1)
    else
        action_log_prob = logpdf.(dist, action)
    end
    EnrichedAction(action; action_log_prob = vec(action_log_prob))
end

function RLBase.update!(
    p::PPOPolicy,
    t::Union{PPOTrajectory,MaskedPPOTrajectory},
    ::AbstractEnv,
    ::PreActStage,
)
    length(t) == 0 && return  # in the first update, only state & action are inserted into trajectory
    p.update_step += 1
    if p.update_step % p.update_freq == 0
        _update!(p, t)
    end
end

function _update!(p::PPOPolicy, t::AbstractTrajectory)
    rng = p.rng
    AC = p.learner
    γ = p.γ
    λ = p.λ
    n_epochs = p.n_epochs
    n_microbatches = p.n_microbatches
    clip_range = p.clip_range
    w₁ = p.actor_loss_weight
    w₂ = p.critic_loss_weight
    w₃ = p.entropy_loss_weight

    n_envs, n_rollout = size(t[:terminal])
    @assert n_envs * n_rollout % n_microbatches == 0 "size mismatch"
    microbatch_size = n_envs * n_rollout ÷ n_microbatches

    n = length(t)
    states_plus = t[:state] |> p.device

    states_flatten = flatten_batch(select_last_dim(states_plus, 1:n))
    states_plus_values =
        reshape(AC.critic(flatten_batch(states_plus)) |> cpu, n_envs, :)
    advantages = generalized_advantage_estimation(
        t[:reward],
        states_plus_values,
        γ,
        λ;
        dims = 2,
        terminal = t[:terminal],
    )
    returns = advantages .+ select_last_dim(states_plus_values, 1:n_rollout)

    actions_flatten = flatten_batch(select_last_dim(t[:action], 1:n))
    action_log_probs = select_last_dim(t[:action_log_prob], 1:n)

    # TODO: normalize advantage
    for epoch in 1:n_epochs
        rand_inds = shuffle!(rng, Vector(1:n_envs*n_rollout))
        for i in 1:n_microbatches
            inds = rand_inds[(i-1)*microbatch_size+1:i*microbatch_size]
            if t isa MaskedPPOTrajectory
                lam = select_last_dim(flatten_batch(select_last_dim(t[:legal_actions_mask], 1:n)),
                        inds,
                    ) |> p.device
                @error "TODO:"
            end
            s = select_last_dim(states_flatten, inds) |> p.device  # !!! performance critical
            a = select_last_dim(actions_flatten, inds) |> p.device
            r = vec(returns)[inds] |> p.device
            log_p = vec(action_log_probs)[inds] |> p.device
            adv = vec(advantages)[inds] |> p.device

            ps = Flux.params(AC)
            gs = gradient(ps) do
                v′ = AC.critic(s) |> vec
                if AC.actor isa GaussianNetwork
                    μ, logσ = AC.actor(s)
                    if ndims(a) == 2
                        log_p′ₐ = vec(sum(normlogpdf(μ, exp.(logσ), a), dims = 1))
                    else
                        log_p′ₐ = normlogpdf(μ, exp.(logσ), a)
                    end
                    entropy_loss = mean(size(logσ, 1) * (log(2.0f0π) + 1) .+ sum(logσ; dims = 1)) / 2
                else
                    # actor is assumed to return discrete logits
                    logit′ = AC.actor(s)
                    p′ = softmax(logit′)
                    log_p′ = logsoftmax(logit′)
                    log_p′ₐ = log_p′[CartesianIndex.(a, 1:length(a))]
                    entropy_loss = -sum(p′ .* log_p′) * 1 // size(p′, 2)
                end
                ratio = exp.(log_p′ₐ .- log_p)
                surr1 = ratio .* adv
                surr2 = clamp.(ratio, 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv

                actor_loss = -mean(min.(surr1, surr2))
                critic_loss = mean((r .- v′) .^ 2)
                loss = w₁ * actor_loss + w₂ * critic_loss - w₃ * entropy_loss

                ignore() do
                    p.actor_loss[i, epoch] = actor_loss
                    p.critic_loss[i, epoch] = critic_loss
                    p.entropy_loss[i, epoch] = entropy_loss
                    p.loss[i, epoch] = loss
                end

                loss
            end

            p.norm[i, epoch] = clip_by_global_norm!(gs, ps, p.max_grad_norm)
            update!(AC, gs)
        end
    end
end

function RLBase.update!(
    trajectory::Union{PPOTrajectory,MaskedPPOTrajectory},
    ::PPOPolicy,
    env::MultiThreadEnv,
    ::PreActStage,
    action::EnrichedAction,
)
    push!(
        trajectory;
        state = state(env),
        action = action.action,
        action_log_prob = action.meta.action_log_prob,
    )

    if trajectory isa MaskedPPOTrajectory
        push!(trajectory; legal_actions_mask = legal_action_space_mask(env))
    end
end