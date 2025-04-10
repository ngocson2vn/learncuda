; ModuleID = 'main.cc'
source_filename = "main.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: noinline norecurse optnone uwtable
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  %2 = alloca float*, align 8
  %3 = alloca i32, align 4
  %4 = alloca float*, align 8
  store i32 0, i32* %1, align 4
  %5 = call noalias nonnull i8* @_Znam(i64 12) #3
  %6 = bitcast i8* %5 to float*
  store float* %6, float** %2, align 8
  store i32 0, i32* %3, align 4
  br label %7

7:                                                ; preds = %19, %0
  %8 = load i32, i32* %3, align 4
  %9 = icmp slt i32 %8, 3
  br i1 %9, label %10, label %22

10:                                               ; preds = %7
  %11 = load i32, i32* %3, align 4
  %12 = sitofp i32 %11 to double
  %13 = fmul double 1.000000e+00, %12
  %14 = fptrunc double %13 to float
  %15 = load float*, float** %2, align 8
  %16 = load i32, i32* %3, align 4
  %17 = sext i32 %16 to i64
  %18 = getelementptr inbounds float, float* %15, i64 %17
  store float %14, float* %18, align 4
  br label %19

19:                                               ; preds = %10
  %20 = load i32, i32* %3, align 4
  %21 = add nsw i32 %20, 1
  store i32 %21, i32* %3, align 4
  br label %7

22:                                               ; preds = %7
  %23 = bitcast float** %4 to i8**
  %24 = call i32 @cudaMalloc(i8** %23, i64 12)
  %25 = load float*, float** %4, align 8
  %26 = bitcast float* %25 to i8*
  %27 = load float*, float** %2, align 8
  %28 = bitcast float* %27 to i8*
  %29 = call i32 @cudaMemcpy(i8* %26, i8* %28, i64 12, i32 1)
  %30 = load float*, float** %4, align 8
  call void @_Z13launch_kernelPfi(float* %30, i32 3)
  %31 = call i32 @cudaDeviceSynchronize()
  ret i32 0
}

; Function Attrs: nobuiltin allocsize(0)
declare dso_local nonnull i8* @_Znam(i64) #1

declare dso_local i32 @cudaMalloc(i8**, i64) #2

declare dso_local i32 @cudaMemcpy(i8*, i8*, i64, i32) #2

declare dso_local void @_Z13launch_kernelPfi(float*, i32) #2

declare dso_local i32 @cudaDeviceSynchronize() #2

attributes #0 = { noinline norecurse optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nobuiltin allocsize(0) "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { builtin allocsize(0) }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"Debian clang version 11.0.1-2~deb10u1"}
