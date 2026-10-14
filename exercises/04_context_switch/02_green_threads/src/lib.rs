//! # Green Thread Scheduler (riscv64)
//!
//! In this exercise, you build a simple cooperative (green) thread scheduler on top of context switching.
//! This crate is **riscv64 only**; run with the repo's normal flow (`./check.sh` / `oscamp`) or natively on riscv64.
//!
//! ## Key Concepts
//!
//! - Cooperative vs preemptive scheduling
//! - Thread state: `Ready`, `Running`, `Finished`
//! - `yield_now()`: current thread voluntarily gives up the CPU
//! - Scheduler loop: pick next ready thread and switch to it
//!
//! ## Design
//!
//! Each green thread has its own stack and `TaskContext`. Threads call `yield_now()` to yield.
//! The scheduler round-robins among ready threads. User entry is wrapped by `thread_wrapper`, which
//! calls the entry then marks the thread `Finished` and switches back.

#![cfg(target_arch = "riscv64")]

use core::arch::naked_asm;

/// Per-thread stack size. Slightly larger to avoid overflow under QEMU / test harness.
const STACK_SIZE: usize = 1024 * 128; // 128KB 栈

/// Task context (riscv64); layout must match `01_stack_coroutine::TaskContext` and the asm below.
#[repr(C)]
#[derive(Debug, Default, Clone)]
pub struct TaskContext {
    sp: u64,
    ra: u64,
    s0: u64,
    s1: u64,
    s2: u64,
    s3: u64,
    s4: u64,
    s5: u64,
    s6: u64,
    s7: u64,
    s8: u64,
    s9: u64,
    s10: u64,
    s11: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThreadState {
    Ready,    // 就绪：等待被调度
    Running,  // 运行中：正在占用CPU
    Finished, // 已完成：执行完毕
}

struct GreenThread {
    ctx: TaskContext,
    state: ThreadState,
    _stack: Option<Vec<u8>>,
    /// 用户入口函数，首次调度时取出交给包装函数执行
    entry: Option<extern "C" fn()>,
}

/// 全局变量：新线程的入口函数，调度器设置、thread_wrapper 读取
static mut CURRENT_THREAD_ENTRY: Option<extern "C" fn()> = None;

/// 每个绿色线程的入口包装函数：调用用户入口，执行完后标记完成并切回调度器
extern "C" fn thread_wrapper() {
    // 读取全局变量里的用户入口
    let entry = unsafe { core::ptr::read(&raw const CURRENT_THREAD_ENTRY) };
    if let Some(f) = entry {
        unsafe { CURRENT_THREAD_ENTRY = None }; // 取走后清空，只执行一次
        f(); // 调用真正的用户线程函数
    }
    thread_finished(); // 用户函数返回后，标记线程结束
}

/// Save current callee-saved regs into `old`, load from `new`, then `ret` to `new.ra`.
/// Zero `a0`/`a1` before `ret` so we don't leak pointers into the new context.
///
/// Must be `#[unsafe(naked)]` to prevent the compiler from generating a prologue/epilogue.
#[unsafe(naked)]
unsafe extern "C" fn switch_context(_old: &mut TaskContext, _new: &TaskContext) {
    naked_asm!(
        "sd sp, 0(a0)",
        "sd ra, 8(a0)",
        "sd s0, 16(a0)",
        "sd s1, 24(a0)",
        "sd s2, 32(a0)",
        "sd s3, 40(a0)",
        "sd s4, 48(a0)",
        "sd s5, 56(a0)",
        "sd s6, 64(a0)",
        "sd s7, 72(a0)",
        "sd s8, 80(a0)",
        "sd s9, 88(a0)",
        "sd s10, 96(a0)",
        "sd s11, 104(a0)",
        "ld sp, 0(a1)",
        "ld ra, 8(a1)",
        "ld s0, 16(a1)",
        "ld s1, 24(a1)",
        "ld s2, 32(a1)",
        "ld s3, 40(a1)",
        "ld s4, 48(a1)",
        "ld s5, 56(a1)",
        "ld s6, 64(a1)",
        "ld s7, 72(a1)",
        "ld s8, 80(a1)",
        "ld s9, 88(a1)",
        "ld s10, 96(a1)",
        "ld s11, 104(a1)",
        "li a0, 0",
        "li a1, 0",
        "ret",
    );
}

pub struct Scheduler {
    threads: Vec<GreenThread>, // 所有线程列表
    current: usize,            // 当前正在运行的线程索引
}

impl Scheduler {
    pub fn new() -> Self {
        // 初始化主线程（索引0），不需要额外栈，用程序默认栈
        let main_thread = GreenThread {
            ctx: TaskContext::default(),
            state: ThreadState::Running,
            _stack: None,
            entry: None,
        };

        Self {
            threads: vec![main_thread],
            current: 0,
        }
    }

    /// Register a new green thread that will run `entry` when first scheduled.
    ///
    /// 1. Allocate a stack of `STACK_SIZE` bytes; compute `stack_top` (high address).
    /// 2. Set up the context: `ra = thread_wrapper` so the first switch jumps to the wrapper;
    ///    `sp` must be 16-byte aligned (e.g. `(stack_top - 16) & !15` to leave headroom).
    /// 3. Push a `GreenThread` with this context, state `Ready`, and `entry` stored for the wrapper to call.
    pub fn spawn(&mut self, entry: extern "C" fn()) {
        // 1. 分配独立栈空间
        let stack = vec![0u8; STACK_SIZE];
        let stack_top = stack.as_ptr() as usize + STACK_SIZE;

        // 2. 初始化任务上下文
        let mut ctx = TaskContext::default();
        // 首次切换先执行包装函数 thread_wrapper
        ctx.ra = thread_wrapper as u64;
        // 栈指针对齐，预留16字节防止栈溢出
        ctx.sp = ((stack_top - 16) & !15) as u64;

        // 3. 创建绿色线程，加入调度队列，状态为就绪
        let thread = GreenThread {
            ctx,
            state: ThreadState::Ready,
            _stack: Some(stack),
            entry: Some(entry),
        };
        self.threads.push(thread);
    }

    /// Run the scheduler until all threads (except the main one) are `Finished`.
    ///
    /// 1. Set the global `SCHEDULER` pointer to `self` so that `yield_now` and `thread_finished` can call back.
    /// 2. Loop: if all threads in `threads[1..]` are `Finished`, break; otherwise call `schedule_next()` (which may switch away and later return).
    /// 3. Clear `SCHEDULER` when done.
    pub fn run(&mut self) {
        // 1. 设置全局调度器指针，让 yield_now 等函数能找到调度器
        unsafe {
            SCHEDULER = self as *mut Scheduler;
        }

        // 2. 调度主循环：直到所有用户线程都执行完
        loop {
            // 检查所有非主线程是否都已结束
            let all_finished = self.threads[1..]
                .iter()
                .all(|t| t.state == ThreadState::Finished);
            if all_finished {
                break;
            }
            // 调度下一个就绪线程
            self.schedule_next();
        }

        // 3. 清空全局指针，避免悬空指针
        unsafe {
            SCHEDULER = std::ptr::null_mut();
        }
    }

    /// Find the next ready thread (starting from `current + 1` round-robin), mark current as `Ready` (if not `Finished`), mark next as `Running`, set `CURRENT_THREAD_ENTRY` if the next thread has an entry, then switch to it.
    fn schedule_next(&mut self) {
        let thread_count = self.threads.len();
        let current_idx = self.current;

        // 1. 轮询查找下一个就绪线程（Round-Robin 轮转调度）
        let mut next_idx = None;
        for offset in 1..=thread_count {
            let idx = (current_idx + offset) % thread_count;
            if self.threads[idx].state == ThreadState::Ready {
                next_idx = Some(idx);
                break;
            }
        }
        // 没有就绪线程就直接返回
        let next_idx = match next_idx {
            Some(idx) => idx,
            None => return,
        };

        // 2. 更新当前线程状态：没结束就放回就绪队列
        if self.threads[current_idx].state != ThreadState::Finished {
            self.threads[current_idx].state = ThreadState::Ready;
        }

        // 3. 下一个线程标记为运行中
        self.threads[next_idx].state = ThreadState::Running;

        // 4. 首次调度的线程，把入口放到全局变量供包装函数读取
        if let Some(entry) = self.threads[next_idx].entry.take() {
            unsafe {
                CURRENT_THREAD_ENTRY = Some(entry);
            }
        }

        // 5. 更新当前线程索引
        self.current = next_idx;

        // 6. 执行上下文切换（核心！）
        unsafe {
            let old_ctx = &mut self.threads[current_idx].ctx;
            let new_ctx = &self.threads[next_idx].ctx;
            switch_context(old_ctx, new_ctx);
        }
    }
}

impl TaskContext {
    fn as_mut_ptr(&mut self) -> *mut TaskContext {
        self as *mut TaskContext
    }
    fn as_ptr(&self) -> *const TaskContext {
        self as *const TaskContext
    }
}

static mut SCHEDULER: *mut Scheduler = std::ptr::null_mut();

/// 当前线程主动让出CPU，调度器切换到下一个就绪线程
pub fn yield_now() {
    unsafe {
        if !SCHEDULER.is_null() {
            (*SCHEDULER).schedule_next();
        }
    }
}

/// 标记当前线程已结束，切换到下一个线程（由 thread_wrapper 自动调用）
fn thread_finished() {
    unsafe {
        if !SCHEDULER.is_null() {
            let sched = &mut *SCHEDULER;
            sched.threads[sched.current].state = ThreadState::Finished;
            sched.schedule_next();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Mutex;

    /// 测试必须串行执行：调度器使用了全局状态
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    static EXEC_ORDER: AtomicU32 = AtomicU32::new(0);

    extern "C" fn task_a() {
        EXEC_ORDER.fetch_add(1, Ordering::SeqCst);
        yield_now();
        EXEC_ORDER.fetch_add(10, Ordering::SeqCst);
        yield_now();
        EXEC_ORDER.fetch_add(100, Ordering::SeqCst);
    }

    extern "C" fn task_b() {
        EXEC_ORDER.fetch_add(1, Ordering::SeqCst);
        yield_now();
        EXEC_ORDER.fetch_add(10, Ordering::SeqCst);
    }

    #[test]
    fn test_scheduler_runs_all() {
        let _guard = TEST_LOCK.lock().unwrap();
        EXEC_ORDER.store(0, Ordering::SeqCst);

        let mut sched = Scheduler::new();
        sched.spawn(task_a);
        sched.spawn(task_b);
        sched.run();

        let got = EXEC_ORDER.load(Ordering::SeqCst);
        if got != 122 {
            panic!(
                "EXEC_ORDER: expected 122, got {} (run with --nocapture to see stderr)",
                got
            );
        }
    }

    static SIMPLE_FLAG: AtomicU32 = AtomicU32::new(0);

    extern "C" fn simple_task() {
        SIMPLE_FLAG.store(42, Ordering::SeqCst);
    }

    #[test]
    fn test_single_thread() {
        let _guard = TEST_LOCK.lock().unwrap();
        SIMPLE_FLAG.store(0, Ordering::SeqCst);

        let mut sched = Scheduler::new();
        sched.spawn(simple_task);
        sched.run();

        assert_eq!(SIMPLE_FLAG.load(Ordering::SeqCst), 42);
    }
}